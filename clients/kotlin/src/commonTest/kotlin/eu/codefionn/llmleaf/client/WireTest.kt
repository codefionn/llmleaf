package eu.codefionn.llmleaf.client

import eu.codefionn.llmleaf.client.model.ChatMessage
import eu.codefionn.llmleaf.client.model.ChatRequest
import eu.codefionn.llmleaf.client.model.MessageContent
import eu.codefionn.llmleaf.client.model.Role
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

private val jsonHeaders = headersOf(HttpHeaders.ContentType, "application/json")

class WireTest {
    @Test
    fun chatRequestSerialisesPlainContentAsString() = runTest {
        val req = ChatRequest(
            model = "gpt-4o-mini",
            messages = listOf(ChatMessage.user("hi")),
        )
        val body = LenientJson.encodeToString(ChatRequest.serializer(), req).let {
            Json.parseToJsonElement(it).jsonObject
        }
        assertEquals("gpt-4o-mini", body["model"]!!.jsonPrimitive.content)
        val msg = body["messages"]!!.let { (it as JsonArray)[0] }.jsonObject
        assertEquals("user", msg["role"]!!.jsonPrimitive.content)
        // content is a bare string, not an array or object.
        assertEquals("hi", msg["content"]!!.jsonPrimitive.content)
        // absent optionals stay off the wire.
        assertNull(body["temperature"])
        assertNull(body["stream"])
    }

    @Test
    fun extraKeysMergeAtTopLevel() = runTest {
        val req = ChatRequest(
            model = "m",
            messages = listOf(ChatMessage.user("x")),
            extra = RawJson("""{"provider":{"order":["a"]},"temperature":0.5}"""),
            temperature = 0.1f,
        )
        val body = Json.parseToJsonElement(
            LenientJson.encodeToString(ChatRequest.serializer(), req),
        ).jsonObject
        // extra's nested object is spliced at the top level, not nested under "extra".
        assertNull(body["extra"])
        assertTrue(body.containsKey("provider"))
        // explicit field wins over an extra of the same key.
        assertEquals(0.1f, body["temperature"]!!.jsonPrimitive.content.toFloat())
    }

    @Test
    fun stopSerialisesBareStringForOneElement() = runTest {
        val one = ChatRequest("m", listOf(ChatMessage.user("x")), stop = listOf("END"))
        val body = Json.parseToJsonElement(
            LenientJson.encodeToString(ChatRequest.serializer(), one),
        ).jsonObject
        assertTrue(body["stop"] is JsonPrimitive)
        assertEquals("END", body["stop"]!!.jsonPrimitive.content)

        val many = ChatRequest("m", listOf(ChatMessage.user("x")), stop = listOf("A", "B"))
        val body2 = Json.parseToJsonElement(
            LenientJson.encodeToString(ChatRequest.serializer(), many),
        ).jsonObject
        assertTrue(body2["stop"] is JsonArray)
    }

    @Test
    fun nonStreamingChatRoundTrips() = runTest {
        val engine = MockEngine { request ->
            assertEquals("Bearer test", request.headers[HttpHeaders.Authorization])
            respond(
                content = """
                    {"id":"c1","object":"chat.completion","created":1,"model":"m",
                     "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},
                     "finish_reason":"stop"}],
                     "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}
                """.trimIndent(),
                status = HttpStatusCode.OK,
                headers = jsonHeaders,
            )
        }
        val client = LlmleafClient("https://gw.example.com", "test", engine)
        val resp = client.chat(ChatRequest("m", listOf(ChatMessage.user("hi"))))
        assertEquals("c1", resp.id)
        val content = resp.choices.single().message.content
        assertTrue(content is MessageContent.Text && content.text == "hello")
        assertEquals(3, resp.usage!!.totalTokens)
        client.close()
    }

    @Test
    fun streamingChatStopsAtDoneSentinel() = runTest {
        val sse = buildString {
            append("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"He\"}}]}\n\n")
            append("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"llo\"}}]}\n\n")
            append("data: [DONE]\n\n")
        }
        val engine = MockEngine {
            respond(sse, HttpStatusCode.OK, headersOf(HttpHeaders.ContentType, "text/event-stream"))
        }
        val client = LlmleafClient("https://gw.example.com", "test", engine)
        val chunks = client.chatStream(ChatRequest("m", listOf(ChatMessage.user("hi")))).toList()
        assertEquals(2, chunks.size) // the [DONE] sentinel is not a chunk
        val text = chunks.joinToString("") { it.choices.first().delta.content ?: "" }
        assertEquals("Hello", text)
        client.close()
    }

    @Test
    fun nonSuccessThrowsApiError() = runTest {
        val engine = MockEngine {
            respond(
                """{"error":{"message":"no route for model"}}""",
                HttpStatusCode.NotFound,
                jsonHeaders,
            )
        }
        val client = LlmleafClient("https://gw.example.com", "test", engine)
        val err = assertFailsWith<ApiError> {
            client.chat(ChatRequest("nope", listOf(ChatMessage.user("hi"))))
        }
        assertEquals(404, err.status)
        assertEquals("no route for model", err.message)
        client.close()
    }

    @Test
    fun base64EmbeddingsDecodeToFloats() = runTest {
        // 1.0f and 2.0f as little-endian f32 bytes, base64-encoded:
        //   1.0f = 00 00 80 3F, 2.0f = 00 00 00 40  ->  AACAPwAAAEA=
        val engine = MockEngine {
            respond(
                """{"object":"list","model":"emb","data":[{"object":"embedding","index":0,"embedding":"AACAPwAAAEA="}],"usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}""",
                HttpStatusCode.OK,
                jsonHeaders,
            )
        }
        val client = LlmleafClient("https://gw.example.com", "test", engine)
        val resp = client.embeddings(
            eu.codefionn.llmleaf.client.model.EmbeddingRequest(
                model = "emb",
                input = listOf("hello"),
                encodingFormat = "base64",
            ),
        )
        assertEquals(listOf(1.0f, 2.0f), resp.data.single().embedding)
        client.close()
    }

    @Test
    fun roleEnumMapsToLowercaseToken() {
        assertEquals("assistant", Role.ASSISTANT.wire)
        val decoded = LenientJson.decodeFromString(Role.serializer(), "\"tool\"")
        assertEquals(Role.TOOL, decoded)
    }
}
