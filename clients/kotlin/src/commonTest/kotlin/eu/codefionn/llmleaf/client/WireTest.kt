package eu.codefionn.llmleaf.client

import eu.codefionn.llmleaf.client.model.ChatMessage
import eu.codefionn.llmleaf.client.model.ChatRequest
import eu.codefionn.llmleaf.client.model.FinishReason
import eu.codefionn.llmleaf.client.model.MessageContent
import eu.codefionn.llmleaf.client.model.ReasoningDetail
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
import kotlin.test.assertFalse
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
    fun streamingChatPreservesSplitToolCallDeltas() = runTest {
        val sse = buildString {
            append("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Par\"}}]}}]}\n\n")
            append("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"is\\\"}\"}}]}}]}\n\n")
            append("data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n")
            append("data: [DONE]\n\n")
        }
        val engine = MockEngine {
            respond(sse, HttpStatusCode.OK, headersOf(HttpHeaders.ContentType, "text/event-stream"))
        }
        val client = LlmleafClient("https://gw.example.com", "test", engine)
        val chunks = client.chatStream(ChatRequest("m", listOf(ChatMessage.user("weather?")))).toList()

        val calls = chunks.flatMap { chunk ->
            chunk.choices.flatMap { choice -> choice.delta.toolCalls }
        }
        assertEquals(2, calls.size)
        assertEquals("call_1", calls[0].id)
        assertEquals("function", calls[0].type)
        assertEquals("get_weather", calls[0].function?.name)
        assertEquals("{\"city\":\"Paris\"}", calls.joinToString("") { it.function?.arguments ?: "" })
        assertEquals(FinishReason.TOOL_CALLS, chunks.last().choices.single().finishReason)
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
    fun rerankRequestSerialisesDocumentsAsArray() = runTest {
        val req = eu.codefionn.llmleaf.client.model.RerankRequest(
            model = "rr",
            query = "the query",
            documents = listOf("only one"),
            topN = 3,
        )
        val body = Json.parseToJsonElement(
            LenientJson.encodeToString(
                eu.codefionn.llmleaf.client.model.RerankRequest.serializer(),
                req,
            ),
        ).jsonObject
        assertEquals("rr", body["model"]!!.jsonPrimitive.content)
        assertEquals("the query", body["query"]!!.jsonPrimitive.content)
        // documents is always an array, even for a single element (no bare-string shortcut).
        assertTrue(body["documents"] is JsonArray)
        assertEquals("only one", (body["documents"] as JsonArray).single().jsonPrimitive.content)
        assertEquals(3, body["top_n"]!!.jsonPrimitive.content.toInt())
        // absent optionals stay off the wire.
        assertNull(body["return_documents"])
    }

    @Test
    fun rerankResultsDecode() = runTest {
        val engine = MockEngine {
            respond(
                """{"object":"list","model":"rr","results":[{"index":2,"relevance_score":0.87,"document":"doc c"},{"index":0,"relevance_score":0.42}],"usage":{"total_tokens":5,"cost_usd":0.001}}""",
                HttpStatusCode.OK,
                jsonHeaders,
            )
        }
        val client = LlmleafClient("https://gw.example.com", "test", engine)
        val resp = client.rerank(
            eu.codefionn.llmleaf.client.model.RerankRequest(
                model = "rr",
                query = "the query",
                documents = listOf("doc a", "doc b", "doc c"),
                topN = 2,
                returnDocuments = true,
            ),
        )
        assertEquals("rr", resp.model)
        assertEquals(2, resp.results.size)
        val top = resp.results.first()
        assertEquals(2, top.index)
        assertEquals(0.87, top.relevanceScore)
        // `document` is surfaced as a raw JsonElement (here a bare string).
        assertEquals("doc c", top.document!!.jsonPrimitive.content)
        // absent `document` decodes to null.
        assertNull(resp.results[1].document)
        assertEquals(5, resp.usage!!.totalTokens)
        assertEquals(0.001, resp.usage!!.costUsd)
        client.close()
    }

    @Test
    fun reasoningAndCachedTokensRoundTrip() = runTest {
        // Response carries flat `reasoning`, an OPEN signed text block and a HIDDEN encrypted
        // block in `reasoning_details`, plus prompt-cache hit (`prompt_tokens_details.cached_tokens`)
        // and write (`cache_creation_tokens`) accounting.
        val engine = MockEngine {
            respond(
                content = """
                    {"id":"c1","object":"chat.completion","created":1,"model":"m",
                     "choices":[{"index":0,"message":{"role":"assistant","content":"hi",
                       "reasoning":"let me think",
                       "reasoning_details":[
                         {"type":"reasoning.text","text":"step one","signature":"sig-abc","index":0},
                         {"type":"reasoning.encrypted","data":"OPAQUE==","format":"anthropic-claude-v1","index":1}
                       ]},
                       "finish_reason":"stop"}],
                     "usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,
                       "prompt_tokens_details":{"cached_tokens":7},"cache_creation_tokens":3}}
                """.trimIndent(),
                status = HttpStatusCode.OK,
                headers = jsonHeaders,
            )
        }
        val client = LlmleafClient("https://gw.example.com", "test", engine)
        val resp = client.chat(ChatRequest("m", listOf(ChatMessage.user("hi"))))
        val msg = resp.choices.single().message
        assertEquals("let me think", msg.reasoning)
        assertEquals(2, msg.reasoningDetails.size)

        val open = msg.reasoningDetails[0]
        assertEquals("reasoning.text", open.type)
        assertEquals("step one", open.text)
        assertEquals("sig-abc", open.signature)
        assertFalse(open.isHidden)
        assertEquals("step one", open.openText)

        val hidden = msg.reasoningDetails[1]
        assertEquals("reasoning.encrypted", hidden.type)
        assertEquals("OPAQUE==", hidden.data)
        assertEquals("anthropic-claude-v1", hidden.format)
        assertTrue(hidden.isHidden)
        assertNull(hidden.openText)

        // Usage cache accounting.
        assertEquals(7, resp.usage!!.cachedTokens)
        assertEquals(7, resp.usage!!.promptTokensDetails!!.cachedTokens)
        assertEquals(3, resp.usage!!.cacheWrites)
        assertEquals(3, resp.usage!!.cacheCreationTokens)
        client.close()
    }

    @Test
    fun assistantReasoningDetailsSerialiseOnRequest() = runTest {
        // Echoing a prior assistant turn back: `reasoning_details` (with the opaque signature/data
        // round-tripped verbatim) must serialise under the snake_case wire keys; an empty list is
        // omitted entirely.
        val priorTurn = ChatMessage(
            role = Role.ASSISTANT,
            content = MessageContent.text("done"),
            reasoning = "because",
            reasoningDetails = listOf(
                ReasoningDetail(type = "reasoning.text", text = "step one", signature = "sig-abc"),
                ReasoningDetail(type = "reasoning.encrypted", data = "OPAQUE=="),
            ),
        )
        val req = ChatRequest(
            model = "m",
            messages = listOf(ChatMessage.user("hi"), priorTurn, ChatMessage.user("continue")),
        )
        val body = Json.parseToJsonElement(
            LenientJson.encodeToString(ChatRequest.serializer(), req),
        ).jsonObject
        val messages = body["messages"] as JsonArray

        val assistant = messages[1].jsonObject
        assertEquals("because", assistant["reasoning"]!!.jsonPrimitive.content)
        val details = assistant["reasoning_details"] as JsonArray
        assertEquals(2, details.size)
        val first = details[0].jsonObject
        assertEquals("reasoning.text", first["type"]!!.jsonPrimitive.content)
        assertEquals("sig-abc", first["signature"]!!.jsonPrimitive.content)
        // unset optional fields stay off the wire.
        assertNull(first["data"])
        assertNull(first["summary"])
        val second = details[1].jsonObject
        assertEquals("OPAQUE==", second["data"]!!.jsonPrimitive.content)

        // a plain user message carries neither key (empty list / null are omitted).
        val user = messages[0].jsonObject
        assertNull(user["reasoning"])
        assertNull(user["reasoning_details"])
    }

    @Test
    fun roleEnumMapsToLowercaseToken() {
        assertEquals("assistant", Role.ASSISTANT.wire)
        val decoded = LenientJson.decodeFromString(Role.serializer(), "\"tool\"")
        assertEquals(Role.TOOL, decoded)
    }
}
