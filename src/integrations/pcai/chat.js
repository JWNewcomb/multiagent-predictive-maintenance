import { ChatOpenAI } from "@langchain/openai";

// Configuration
// Point this to your local server (e.g., http://localhost:11434/v1 for Ollama)
const LOCAL_API_URL = process.env.OPENAI_BASE_URL || "http://localhost:11434/v1";
const COMPLETION_MODEL = process.env.COMPLETION_MODEL || "llama3";
// Local models don't check the key, but the SDK requires a non-empty string
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "dummy-local-key";

/**
 * Initialize a ChatOpenAI client pointing to the local endpoint
 * @returns {ChatOpenAI} Initialized LangChain client
 */
let llmClient = null;

export function createLLMClient() {
  if (!llmClient) {
    llmClient = new ChatOpenAI({
      modelName: COMPLETION_MODEL, // The model name your local server expects
      openAIApiKey: OPENAI_API_KEY,
      configuration: {
        baseURL: LOCAL_API_URL,
      },
      temperature: 0, // Recommended for deterministic maintenance tasks
    });
  }
  return llmClient;
}

/**
 * Send messages to the local LLM and get a response
 * @param {Array} messages - Array of messages in LangChain format (HumanMessage, SystemMessage, etc.)
 * @returns {Promise<Object>} - Response from the model (AIMessage)
 */
export async function invokeLLM(messages) {
  try {
    const model = createLLMClient();
    return await model.invoke(messages);
  } catch (error) {
    console.error("Error invoking Local LLM:", error);
    throw new Error(`Local LLM conversation failed: ${error.message}`);
  }
}

/**
 * Stream responses from the local LLM
 * @param {Array} messages - Array of messages in LangChain format
 * @returns {Promise<AsyncIterable>} - Stream of responses
 */
export async function streamFromLLM(messages) {
  try {
    const model = createLLMClient();
    return await model.stream(messages);
  } catch (error) {
    console.error("Error streaming from Local LLM:", error);
    throw new Error(`Local LLM streaming failed: ${error.message}`);
  }
}
