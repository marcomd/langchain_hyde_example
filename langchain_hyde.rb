# frozen_string_literal: true

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
  def answer(query, top_k: 3)
    puts "\nProcessing query: #{query}"

    # Step 1: Generate a hypothetical ANSWER (not just a document)
    # This is key to HyDE - we generate what we think the perfect answer would look like
    hypothetical_answer = generate_hypothetical_answer(query)
    puts "\n1. Generated hypothetical answer:"
    puts "--------------------------------"
    puts hypothetical_answer.completion

    # Step 2: Use this hypothetical answer to find similar real documents
    # The embedding of the hypothetical answer helps find truly relevant content
    relevant_docs = hyde_retrieve_documents(hypothetical_answer.completion)
    puts "\n2. Retrieved relevant documents using HyDE:"
    puts "--------------------------------"
    relevant_docs.each_with_index do |doc, i|
      puts "\nDocument #{i + 1}: #{doc[:source]}"
      puts doc[:content]
    end

    # Step 3: Generate final answer using retrieved documents
    final_answer = generate_final_answer(query, relevant_docs)
    puts "\n3. Final answer (grounded in real documents):"
    puts "--------------------------------"
    puts final_answer.completion

    {
      hypothetical_answer: hypothetical_answer.completion,
      retrieved_documents: relevant_docs,
      final_answer: final_answer.completion
    }
  end

  private

  # Generate a hypothetical document based on a query
  # @param query [String] The query to generate a hypothetical document for
  # @return [Langchain::Completion] The hypothetical document
  def generate_hypothetical_answer(query)
    prompt = <<~PROMPT
      Generate a direct, factual answer to this question: "#{query}"
      The answer should be what you'd expect to find in a high-quality document about this topic.
      Focus on specific details and facts that would help identify relevant documents.
      
      Answer:
    PROMPT

    llm.complete(prompt: prompt)
  end

  # Generate a hypothetical document based on a hypothetical answer
  # @param hypothetical_answer [String] The hypothetical answer to generate a document for
  # @return [Array<Float>] Similarity embeddings of the hypothetical answer
  def hyde_retrieve_documents(hypothetical_answer, top_k: 3)
    # Use the hypothetical answer's embedding to find similar real documents
    hyde_embedding = simple_embedding(hypothetical_answer)

    # Find similar documents using the hypothetical answer embedding
    doc_embeddings = @documents.map do |doc|
      {
        id: doc[:id],
        content: doc[:content],
        source: doc[:source],
        embedding: simple_embedding(doc[:content])
      }
    end

    # Calculate similarities using the hypothetical answer as the reference
    similarities = doc_embeddings.map do |doc|
      similarity = cosine_similarity(hyde_embedding, doc[:embedding])
      { similarity: similarity, document: doc }
    end

    similarities
      .sort_by { |item| -item[:similarity] }
      .take(top_k)
      .map { |item| item[:document] }
  end

  # Generate the final answer based on the query and relevant documents
  # @param query [String] The original query
  # @param relevant_docs [Array<Hash>] An array of relevant documents
  # @return [Langchain::Completion] The final answer
  def generate_final_answer(query, relevant_docs)
    context = relevant_docs.map do |doc|
      "Source (#{doc[:source]}): #{doc[:content]}"
    end.join("\n\n")

    prompt = <<~PROMPT
      Question: #{query}

      Using only the information from these sources, provide a well-supported answer:
      #{context}

      Requirements:
      1. Only use information from the provided sources
      2. Cite the sources when making specific claims
      3. If the sources don't fully answer the question, acknowledge this
      
      Answer:
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
    ]
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
def demonstrate_hyde_rag
  puts "Initializing HyDE-RAG system..."
  hyde_retriever = HydeRetriever.new

  query = "What effect does meditation have on the brain and stress levels?"

  puts "\nDemonstrating HyDE-RAG process..."
  puts "=" * 50
  hyde_retriever.answer(query)

rescue StandardError => e
  puts "Error occurred: #{e.message}"
  puts e.backtrace
end

# Run the example if this file is executed directly
if __FILE__ == $0
  demonstrate_hyde_rag
end