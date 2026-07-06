package eu.codefionn.llmleaf.client

import eu.codefionn.llmleaf.client.model.ResponseContent
import eu.codefionn.llmleaf.client.model.ResponseContentPart
import eu.codefionn.llmleaf.client.model.ResponseFunctionCallItem
import eu.codefionn.llmleaf.client.model.ResponseFunctionCallOutputItem
import eu.codefionn.llmleaf.client.model.ResponseMessageItem
import eu.codefionn.llmleaf.client.model.ResponseReasoningItem
import eu.codefionn.llmleaf.client.model.ResponsesInput
import eu.codefionn.llmleaf.client.model.ResponsesRequest
import eu.codefionn.llmleaf.client.model.ResponsesToolChoice
import eu.codefionn.llmleaf.client.model.ResponsesToolDef
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.content.TextContent
import io.ktor.http.headersOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

private val jsonHeaders = headersOf(HttpHeaders.ContentType, "application/json")
private val sseHeaders = headersOf(HttpHeaders.ContentType, "text/event-stream")

class ResponsesWireTest {
    @Test
    fun inputSerialisesAsBareStringForSingleMessage() = runTest {
        val req = ResponsesRequest(model = "gpt-4o-mini", input = ResponsesInput.text("hi"))
        val body = Json.parseToJsonElement(
            LenientJson.encodeToString(ResponsesRequest.serializer(), req),
        ).jsonObject
        assertEquals("gpt-4o-mini", body["model"]!!.jsonPrimitive.content)
        // a bare-string input serialises as a JSON string, not an array.
        assertTrue(body["input"] is JsonPrimitive)
        assertEquals("hi", body["input"]!!.jsonPrimitive.content)
        // absent optionals stay off the wire.
        assertNull(body["stream"])
        assertNull(body["tools"])
    }

    @Test
    fun requestBodySerialisesFlatToolsAndItemArray() = runTest {
        // A full replay turn: user message, the assistant's function_call, our function_call_output,
        // a reasoning item to continue an encrypted turn, then a follow-up user message.
        val req = ResponsesRequest(
            model = "gpt-4o-mini",
            input = ResponsesInput.items(
                ResponseMessageItem.user("What's the weather in Paris?"),
                ResponseFunctionCallItem(
                    callId = "call_1",
                    name = "get_weather",
                    arguments = """{"city":"Paris"}""",
                    id = "fc_1",
                ),
                ResponseFunctionCallOutputItem(callId = "call_1", output = """{"tempC":21}"""),
                ResponseReasoningItem(
                    id = "rs_1",
                    summary = listOf("Looked up Paris weather."),
                    encryptedContent = "ENC==",
                ),
                ResponseMessageItem.user("And in Celsius?"),
            ),
            instructions = "Be concise.",
            tools = listOf(
                ResponsesToolDef(
                    name = "get_weather",
                    description = "Look up the weather",
                    parameters = RawJson("""{"type":"object","properties":{"city":{"type":"string"}}}"""),
                ),
            ),
            toolChoice = ResponsesToolChoice.function("get_weather"),
        )
        val body = Json.parseToJsonElement(
            LenientJson.encodeToString(ResponsesRequest.serializer(), req),
        ).jsonObject

        assertEquals("gpt-4o-mini", body["model"]!!.jsonPrimitive.content)
        assertEquals("Be concise.", body["instructions"]!!.jsonPrimitive.content)

        // Tools are FLAT — type/name/parameters at the top level, no nested `function` object.
        val tool = (body["tools"] as JsonArray).single().jsonObject
        assertEquals("function", tool["type"]!!.jsonPrimitive.content)
        assertEquals("get_weather", tool["name"]!!.jsonPrimitive.content)
        assertEquals("Look up the weather", tool["description"]!!.jsonPrimitive.content)
        // parameters is spliced as a JSON object, not a string.
        assertTrue(tool["parameters"] is JsonObject)
        assertEquals("object", tool["parameters"]!!.jsonObject["type"]!!.jsonPrimitive.content)
        assertNull(tool["function"])

        // tool_choice is the FLAT named object.
        val choice = body["tool_choice"]!!.jsonObject
        assertEquals("function", choice["type"]!!.jsonPrimitive.content)
        assertEquals("get_weather", choice["name"]!!.jsonPrimitive.content)
        assertNull(choice["function"])

        val input = body["input"] as JsonArray
        assertEquals(5, input.size)

        // (0) plain user message: role-keyed, NO "type" key, content a bare string.
        val userMsg = input[0].jsonObject
        assertNull(userMsg["type"])
        assertEquals("user", userMsg["role"]!!.jsonPrimitive.content)
        assertEquals("What's the weather in Paris?", userMsg["content"]!!.jsonPrimitive.content)

        // (1) function_call: typed, flat.
        val call = input[1].jsonObject
        assertEquals("function_call", call["type"]!!.jsonPrimitive.content)
        assertEquals("fc_1", call["id"]!!.jsonPrimitive.content)
        assertEquals("call_1", call["call_id"]!!.jsonPrimitive.content)
        assertEquals("get_weather", call["name"]!!.jsonPrimitive.content)
        assertEquals("""{"city":"Paris"}""", call["arguments"]!!.jsonPrimitive.content)

        // (2) function_call_output.
        val output = input[2].jsonObject
        assertEquals("function_call_output", output["type"]!!.jsonPrimitive.content)
        assertEquals("call_1", output["call_id"]!!.jsonPrimitive.content)
        assertEquals("""{"tempC":21}""", output["output"]!!.jsonPrimitive.content)

        // (3) reasoning: summary entries take the "summary_text" token; encrypted_content echoed;
        // the empty content list is omitted.
        val reasoning = input[3].jsonObject
        assertEquals("reasoning", reasoning["type"]!!.jsonPrimitive.content)
        assertEquals("rs_1", reasoning["id"]!!.jsonPrimitive.content)
        val summary = reasoning["summary"]!!.jsonArray.single().jsonObject
        assertEquals("summary_text", summary["type"]!!.jsonPrimitive.content)
        assertEquals("Looked up Paris weather.", summary["text"]!!.jsonPrimitive.content)
        assertNull(reasoning["content"])
        assertEquals("ENC==", reasoning["encrypted_content"]!!.jsonPrimitive.content)

        // (4) follow-up user message.
        assertEquals("And in Celsius?", input[4].jsonObject["content"]!!.jsonPrimitive.content)
    }

    @Test
    fun inputImageAndOutputTextPartsUseResponsesShapes() = runTest {
        val req = ResponsesRequest(
            model = "m",
            input = ResponsesInput.items(
                ResponseMessageItem(
                    role = "user",
                    content = ResponseContent.parts(
                        ResponseContentPart.InputText("Describe this"),
                        ResponseContentPart.InputImage("https://img.example/a.png", detail = "low"),
                    ),
                ),
            ),
        )
        val body = Json.parseToJsonElement(
            LenientJson.encodeToString(ResponsesRequest.serializer(), req),
        ).jsonObject
        val parts = (body["input"] as JsonArray).single().jsonObject["content"] as JsonArray

        val textPart = parts[0].jsonObject
        assertEquals("input_text", textPart["type"]!!.jsonPrimitive.content)
        assertEquals("Describe this", textPart["text"]!!.jsonPrimitive.content)

        val imagePart = parts[1].jsonObject
        assertEquals("input_image", imagePart["type"]!!.jsonPrimitive.content)
        // image_url is a plain STRING here, not the chat dialect's nested {url} object.
        assertTrue(imagePart["image_url"] is JsonPrimitive)
        assertEquals("https://img.example/a.png", imagePart["image_url"]!!.jsonPrimitive.content)
        assertEquals("low", imagePart["detail"]!!.jsonPrimitive.content)

        // A constructed output_text part emits an empty annotations array.
        val outText = LenientJson.encodeToString(
            ResponseContentPart.serializer(),
            ResponseContentPart.OutputText("done"),
        )
        val outObj = Json.parseToJsonElement(outText).jsonObject
        assertEquals("output_text", outObj["type"]!!.jsonPrimitive.content)
        assertTrue(outObj["annotations"] is JsonArray)
        assertEquals(0, outObj["annotations"]!!.jsonArray.size)
    }

    @Test
    fun extraKeysMergeAtTopLevel() = runTest {
        val req = ResponsesRequest(
            model = "m",
            input = ResponsesInput.text("x"),
            temperature = 0.1f,
            extra = RawJson("""{"service_tier":"flex","temperature":0.9}"""),
        )
        val body = Json.parseToJsonElement(
            LenientJson.encodeToString(ResponsesRequest.serializer(), req),
        ).jsonObject
        // extra is spliced at the top level, not nested under "extra".
        assertNull(body["extra"])
        assertEquals("flex", body["service_tier"]!!.jsonPrimitive.content)
        // explicit field wins over an extra of the same key.
        assertEquals(0.1f, body["temperature"]!!.jsonPrimitive.content.toFloat())
    }

    @Test
    fun requestRoundTripsThroughDecode() = runTest {
        val original = ResponsesRequest(
            model = "m",
            input = ResponsesInput.items(
                ResponseMessageItem.developer("You are terse."),
                ResponseMessageItem.user("hi"),
            ),
            maxOutputTokens = 256,
            store = false,
            extra = RawJson("""{"service_tier":"default"}"""),
        )
        val encoded = LenientJson.encodeToString(ResponsesRequest.serializer(), original)
        val decoded = LenientJson.decodeFromString(ResponsesRequest.serializer(), encoded)
        assertEquals(original.model, decoded.model)
        assertEquals(256, decoded.maxOutputTokens)
        assertEquals(false, decoded.store)
        // unknown top-level keys are captured back into `extra`.
        assertEquals(original.extra, decoded.extra)
        val items = (decoded.input as ResponsesInput.Items).items
        assertEquals(2, items.size)
        assertEquals("developer", (items[0] as ResponseMessageItem).role)
    }

    @Test
    fun nonStreamingResponsesRoundTrips() = runTest {
        var sentBody: String? = null
        val engine = MockEngine { request ->
            assertEquals("Bearer test", request.headers[HttpHeaders.Authorization])
            sentBody = (request.body as TextContent).text
            respond(
                content = """
                    {"id":"resp_1","object":"response","created_at":1720000000,"status":"completed",
                     "model":"gpt-4o-mini","store":false,
                     "output":[
                       {"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"ENC=="},
                       {"type":"message","id":"msg_1","role":"assistant","status":"completed",
                        "content":[{"type":"output_text","text":"Hi there!","annotations":[]}]}
                     ],
                     "usage":{"input_tokens":11,"input_tokens_details":{"cached_tokens":8},
                       "output_tokens":5,"output_tokens_details":{"reasoning_tokens":2},"total_tokens":16}}
                """.trimIndent(),
                status = HttpStatusCode.OK,
                headers = jsonHeaders,
            )
        }
        val client = LlmleafClient("https://gw.example.com", "test", engine)
        val resp = client.responses(ResponsesRequest("gpt-4o-mini", ResponsesInput.text("hi")))

        // Request forced stream=false.
        val sent = Json.parseToJsonElement(sentBody!!).jsonObject
        assertEquals(false, sent["stream"]!!.jsonPrimitive.content.toBoolean())

        assertEquals("resp_1", resp.id)
        assertEquals("response", resp.`object`)
        assertEquals("completed", resp.status)
        // llmleaf is stateless: store is always false.
        assertEquals(false, resp.store)

        // Output decodes the reasoning + message items; assembled text flattens output_text parts.
        assertEquals(2, resp.output.size)
        val reasoning = resp.output[0] as ResponseReasoningItem
        assertEquals("ENC==", reasoning.encryptedContent)
        val msg = resp.output[1] as ResponseMessageItem
        assertEquals("assistant", msg.role)
        assertEquals("completed", msg.status)
        assertEquals("Hi there!", resp.outputText)

        // Usage in the Responses dialect's own names, incl. cached_tokens + reasoning_tokens.
        assertEquals(11, resp.usage!!.inputTokens)
        assertEquals(8, resp.usage!!.cachedTokens)
        assertEquals(2, resp.usage!!.reasoningTokens)
        assertEquals(16, resp.usage!!.totalTokens)
        client.close()
    }

    @Test
    fun streamingResponsesEmitsTypedEventsAndStopsAtTerminal() = runTest {
        // Typed SSE frames (`event:` + `data:`), no `[DONE]` sentinel. Includes an unknown event
        // type that must be skipped, and a terminal `response.completed` carrying the usage snapshot.
        val sse = buildString {
            append("event: response.created\n")
            append("data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"m\"}}\n\n")
            append("event: response.output_item.added\n")
            append("data: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"\"}}\n\n")
            append("event: response.function_call_arguments.delta\n")
            append("data: {\"type\":\"response.function_call_arguments.delta\",\"sequence_number\":2,\"item_id\":\"fc_1\",\"delta\":\"{\\\"city\\\":\\\"Paris\\\"}\"}\n\n")
            // An unknown event type — must be ignored (not emitted).
            append("event: response.telemetry.ping\n")
            append("data: {\"type\":\"response.telemetry.ping\",\"sequence_number\":3}\n\n")
            append("event: response.output_text.delta\n")
            append("data: {\"type\":\"response.output_text.delta\",\"sequence_number\":4,\"item_id\":\"msg_1\",\"delta\":\"He\"}\n\n")
            append("event: response.output_text.delta\n")
            append("data: {\"type\":\"response.output_text.delta\",\"sequence_number\":5,\"item_id\":\"msg_1\",\"delta\":\"llo\"}\n\n")
            append("event: response.completed\n")
            append("data: {\"type\":\"response.completed\",\"sequence_number\":6,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"m\",\"usage\":{\"input_tokens\":9,\"output_tokens\":3,\"total_tokens\":12}}}\n\n")
            // Anything after the terminal event must never be read.
            append("event: response.created\n")
            append("data: {\"type\":\"response.created\",\"sequence_number\":7}\n\n")
        }
        val engine = MockEngine { respond(sse, HttpStatusCode.OK, sseHeaders) }
        val client = LlmleafClient("https://gw.example.com", "test", engine)

        val events = client.responsesStream(
            ResponsesRequest("m", ResponsesInput.text("weather?")),
        ).toList()

        // The unknown type is skipped and nothing past the terminal event is emitted.
        assertEquals(
            listOf(
                "response.created",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.completed",
            ),
            events.map { it.type },
        )
        // sequence numbers stay strictly increasing across the surfaced events.
        assertEquals(listOf(0L, 1L, 2L, 4L, 5L, 6L), events.map { it.sequenceNumber })

        // The output_item.added event decodes its function_call item.
        val added = events[1].item as ResponseFunctionCallItem
        assertEquals("call_1", added.callId)

        // Accumulate the output_text deltas for the assembled text.
        val text = events.filter { it.type == "response.output_text.delta" }.joinToString("") { it.delta ?: "" }
        assertEquals("Hello", text)

        // The terminal event carries the usage snapshot.
        val terminal = events.last()
        assertEquals("response.completed", terminal.type)
        assertEquals(12, terminal.response!!.usage!!.totalTokens)
        client.close()
    }

    @Test
    fun streamingErrorEventSurfacesAsApiError() = runTest {
        val sse = buildString {
            append("event: response.created\n")
            append("data: {\"type\":\"response.created\",\"sequence_number\":0}\n\n")
            append("event: error\n")
            append("data: {\"type\":\"error\",\"sequence_number\":1,\"message\":\"upstream exploded\"}\n\n")
        }
        val engine = MockEngine { respond(sse, HttpStatusCode.OK, sseHeaders) }
        val client = LlmleafClient("https://gw.example.com", "test", engine)
        val err = assertFailsWith<ApiError> {
            client.responsesStream(ResponsesRequest("m", ResponsesInput.text("x"))).toList()
        }
        assertEquals("upstream exploded", err.message)
        client.close()
    }

    @Test
    fun nonSuccessThrowsApiError() = runTest {
        val engine = MockEngine {
            respond(
                """{"error":{"message":"background: true is not supported"}}""",
                HttpStatusCode.BadRequest,
                jsonHeaders,
            )
        }
        val client = LlmleafClient("https://gw.example.com", "test", engine)
        val err = assertFailsWith<ApiError> {
            client.responses(ResponsesRequest("m", ResponsesInput.text("x")))
        }
        assertEquals(400, err.status)
        assertEquals("background: true is not supported", err.message)
        client.close()
    }
}
