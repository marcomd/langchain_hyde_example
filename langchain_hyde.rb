require 'langchain'
require 'matrix'
require 'debug'
require 'faraday'

class HydeRetriever
  attr_reader :llm, :documents

  # Choose a model here https://github.com/ollama/ollama?tab=readme-ov-file#model-library
  MODEL = 'llama3.2'

  def initialize
    # Initialize Ollama LLM (make sure you have Ollama running locally)
    @llm ||= Langchain::LLM::Ollama.new(
      url: 'http://localhost:11434',
      default_options: { temperature: 0.1, chat_model: MODEL, completion_model: MODEL, embedding_model: MODEL }
    )
    @documents = load_sample_documents
  end

  # Retrieve relevant documents for a given query
  # @param query [String] The query to retrieve documents for
  # @param top_k [Integer] The number of documents to retrieve
  # @return [Array<Hash>] An array of relevant documents
  def retrieve(query, top_k: 3)
    # Step 1: Generate hypothetical document
    hypothetical_doc = generate_hypothetical_document(query)
    puts "\nGenerated Hypothetical Document:"
    puts "--------------------------------"
    puts hypothetical_doc.completion

    # Step 2: Generate simple embedding for hypothetical document
    hyde_embedding = simple_embedding(hypothetical_doc.completion)

    # Step 3: Find similar documents
    similar_docs = find_similar_documents(hyde_embedding, top_k)

    similar_docs
  end

  private

  # Generate a hypothetical document based on a query
  # @param query [String] The query to generate a document for
  # @return [Langchain::LLM::Completion] The generated document
  def generate_hypothetical_document(query)
    prompt = <<~PROMPT
      Generate a detailed, factual document that would be a perfect answer to this query: "#{query}"
      The document should be comprehensive but concise, focusing on verified information.
      Do not include personal opinions or unverified claims.
      Keep the response under 300 words.
    PROMPT

    llm.complete(prompt: prompt)
  end

  # Generate a simple embedding for a given text
  # @param text [String] The text to generate an embedding for
  # @return [Array<Float>] The embedding of the text
  def simple_embedding(text)
    # A simple TF-IDF like embedding approach
    # In production, you'd want to use a proper embedding model
    words = text.downcase.gsub(/[^a-z0-9\s]/, '').split
    unique_words = words.uniq

    # Create a basic numerical representation
    unique_words.map do |word|
      frequency = words.count(word)
      frequency.to_f / words.length
    end
  end

  # Load sample documents for demonstration
  # @return [Array<Hash>] An array of sample documents
  def load_sample_documents
    [
      {
        id: 1,
        content: "Scientific studies have shown that regular meditation practice reduces stress levels significantly. Research indicates a 30% decrease in cortisol levels among regular practitioners. Additionally, MRI scans show increased activity in areas of the brain associated with focus and emotional regulation.",
        source: "Journal of Meditation Studies, 2023"
      },
      {
        id: 2,
        content: "Regular meditation has been linked to improved sleep quality. A study of 500 participants showed that those who meditated for 20 minutes before bed fell asleep 15 minutes faster on average and reported better sleep quality scores.",
        source: "Sleep Research Institute, 2022"
      },
      {
        id: 3,
        content: "Long-term meditation practitioners demonstrate enhanced immune system function. Research shows increased levels of antibodies and improved inflammatory responses in those who meditate regularly.",
        source: "Immunity & Health Journal, 2023"
      },
      {
        id: 4,
        content: "Meditation's impact on anxiety and depression has been well-documented. Clinical trials show it can be as effective as some medications for mild to moderate cases, particularly when combined with traditional therapy.",
        source: "Mental Health Research Quarterly, 2023"
      },
      # This document is not relevant to the query
      {
        id: 5,
        content: "The history of puncakes dates back to ancient Greece, where they were made with wheat flour, olive oil, honey, and curdled milk. The modern pancake recipe with baking powder was developed in the 19th century.",
        source: "Historical Cooking Journal, 2021"
      }
    ]
  end

  # Find similar documents to a given query embedding
  # @param query_embedding [Array<Float>] The embedding of the query
  # @param top_k [Integer] The number of similar documents to return
  # @return [Array<Hash>] An array of similar documents
  def find_similar_documents(query_embedding, top_k)
    # Calculate embeddings for all documents
    doc_embeddings = documents.map do |doc|
      {
        id: doc[:id],
        content: doc[:content],
        source: doc[:source],
        embedding: simple_embedding(doc[:content])
      }
    end

    # Calculate cosine similarity between query and all documents
    similarities = doc_embeddings.map do |doc|
      similarity = cosine_similarity(query_embedding, doc[:embedding])
      { similarity: similarity, document: doc }
    end

    # Sort by similarity and return top k results
    similarities
      .sort_by { |item| -item[:similarity] }
      .take(top_k)
      .map { |item| item[:document] }
  end

  # Calculate cosine similarity between two vectors
  # @param vec1 [Array<Float>] The first vector
  # @param vec2 [Array<Float>] The second vector
  # @return [Float] The cosine similarity between the two vectors
  def cosine_similarity(vec1, vec2)
    # Ensure vectors are of equal length by padding with zeros
    max_length = [vec1.length, vec2.length].max
    v1 = Vector.elements(vec1 + [0] * (max_length - vec1.length))
    v2 = Vector.elements(vec2 + [0] * (max_length - vec2.length))

    # Calculate cosine similarity
    dot_product = v1.inner_product(v2)
    magnitude_product = v1.magnitude * v2.magnitude

    # Avoid division by zero
    return 0.0 if magnitude_product.zero?
    dot_product / magnitude_product
  end
end

# Run the example
# This example demonstrates how to use the HyDE retriever to retrieve relevant documents for a given query
def run_meditation_example
  # Initialize the HyDE retriever
  puts "Initializing HyDE retriever..."
  retriever = HydeRetriever.new

  # Query about meditation benefits
  query = "What are the health benefits of meditation?"
  puts "\nQuery: #{query}"
  puts "\nRetrieving relevant documents..."

  # Get relevant documents
  results = retriever.retrieve(query)

  puts "\nRetrieved Documents:"
  puts "-------------------"
  results.each_with_index do |doc, index|
    puts "\n#{index + 1}. Source: #{doc[:source]}"
    puts "   Content: #{doc[:content]}"
  end
rescue StandardError => e
  puts "Error occurred: #{e.message}"
  puts e.backtrace
end

# Run the example if this file is executed directly
if __FILE__ == $0
  run_meditation_example
end