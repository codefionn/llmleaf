package eu.codefionn.llmleaf.example

import eu.codefionn.llmleaf.client.ApiError
import eu.codefionn.llmleaf.client.LlmleafClient
import eu.codefionn.llmleaf.client.model.ChatMessage
import eu.codefionn.llmleaf.client.model.ChatRequest
import eu.codefionn.llmleaf.client.model.MessageContent
import eu.codefionn.llmleaf.client.model.ResponsesInput
import eu.codefionn.llmleaf.client.model.ResponsesRequest
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking

/**
 * Minimal end-to-end demo of the llmleaf KMP client. Reads its target from the environment:
 *
 *   LLMLEAF_BASE_URL   gateway base URL (default http://localhost:8080)
 *   LLMLEAF_API_KEY    bearer key (required)
 *   LLMLEAF_MODEL      model id (default gpt-4o-mini)
 *
 * Run with:  ./gradlew :example:run
 */
fun main(): Unit = runBlocking {
    val baseUrl = System.getenv("LLMLEAF_BASE_URL") ?: "http://localhost:8080"
    val apiKey = System.getenv("LLMLEAF_API_KEY")
        ?: error("set LLMLEAF_API_KEY to your gateway key")
    val model = System.getenv("LLMLEAF_MODEL") ?: "gpt-4o-mini"

    val client = LlmleafClient(baseUrl, apiKey) {
        // adminToken = System.getenv("LLMLEAF_ADMIN_TOKEN")
    }

    try {
        // 1) List models.
        println("== models ==")
        val models = client.listModels()
        models.data.take(10).forEach { println("  ${it.id}") }
        if (models.data.size > 10) println("  ... and ${models.data.size - 10} more")

        // 2) Non-streaming chat: print the assembled text.
        println("\n== chat (non-streaming) ==")
        val resp = client.chat(
            ChatRequest(
                model = model,
                messages = listOf(
                    ChatMessage.system("You are concise."),
                    ChatMessage.user("Say hello in one short sentence."),
                ),
            ),
        )
        val text = resp.choices.firstOrNull()?.message?.content
        when (text) {
            is MessageContent.Text -> println("  ${text.text}")
            else -> println("  (no text content)")
        }
        resp.usage?.let { println("  usage: ${it.totalTokens} tokens, cost=${it.costUsd ?: "n/a"}") }

        // 3) Streaming chat: collect the Flow and print deltas as they arrive.
        println("\n== chat (streaming) ==")
        print("  ")
        client.chatStream(
            ChatRequest(
                model = model,
                messages = listOf(ChatMessage.user("Count from 1 to 5.")),
            ),
        ).collect { chunk ->
            chunk.choices.firstOrNull()?.delta?.content?.let { print(it) }
        }
        println()

        // 4) Responses dialect (non-streaming): assembled text + Responses-named usage.
        println("\n== responses (non-streaming) ==")
        val respResp = client.responses(
            ResponsesRequest(
                model = model,
                input = ResponsesInput.text("Say hello in one short sentence."),
                instructions = "You are concise.",
            ),
        )
        println("  ${respResp.outputText}")
        respResp.usage?.let { println("  usage: ${it.totalTokens} tokens (cached=${it.cachedTokens})") }

        // 5) Responses dialect (streaming): typed events, no [DONE] sentinel — accumulate the
        //    output_text deltas and stop on the terminal event.
        println("\n== responses (streaming) ==")
        print("  ")
        client.responsesStream(
            ResponsesRequest(
                model = model,
                input = ResponsesInput.text("Count from 1 to 5."),
            ),
        ).collect { event ->
            if (event.type == "response.output_text.delta") print(event.delta ?: "")
        }
        println()
    } catch (e: ApiError) {
        System.err.println("API error ${e.status}: ${e.message}")
    } finally {
        client.close()
    }
}
