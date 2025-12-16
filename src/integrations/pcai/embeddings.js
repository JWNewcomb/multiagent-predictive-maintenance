import OpenAI from "openai";

// Configuration
// For local setups (like Ollama), the API Key is often ignored but required by the SDK type check.
const LOCAL_API_URL = process.env.OPENAI_BASE_URL || "http://localhost:11434/v1"; 
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-3-small";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "dummy-key";

let openAIClient = null;

function getOpenAIClient() {
  if (!openAIClient) {
    openAIClient = new OpenAI({
      baseURL: LOCAL_API_URL,
      apiKey: OPENAI_API_KEY, 
    });
  }
  return openAIClient;
}

/**
 * Generate an embedding for a given text using a local OpenAI-compatible API.
 * @param {string} text - The text to embed.
 * @returns {Promise<Array<number>>} The embedding vector.
 */
export async function generateEmbedding(text) {
  try {
    const client = getOpenAIClient();
    
    const response = await client.embeddings.create({
      model: EMBEDDING_MODEL,
      input: text,
      encoding_format: "float",
    });

    // OpenAI response structure: response.data[0].embedding
    return response.data[0].embedding;
  } catch (error) {
    console.error("Error generating embedding:", error);
    throw error;
  }
}
