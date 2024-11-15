# frozen_string_literal: true

require 'bundler/setup'
require 'pg'
require 'pgvector'
require 'langchain'
require 'matrix'
require 'debug'
require 'faraday'
require 'csv'
require 'nokogiri'
require 'pdf-reader'
require 'docx'

class HydeRetriever
  attr_reader :llm, :documents, :vector_client, :vector_table, :root_path

  # Choose a model here https://github.com/ollama/ollama?tab=readme-ov-file#model-library
  MODEL = 'llama3.2'

  # Initialize the HyDE retriever
  # @param erase [Boolean] Whether to erase existing vector DB data
  def initialize(erase:)
    # Initialize Ollama LLM (make sure you have Ollama running locally)
    @llm ||= Langchain::LLM::Ollama.new(
      url: 'http://localhost:11434',
      default_options: { temperature: 0.1, chat_model: MODEL, completion_model: MODEL, embedding_model: MODEL }
    )
    @vector_client = Langchain::Vectorsearch::Pgvector.new(
      url: 'postgres://postgres:postgres@localhost:5432/langchain_hyde',
      index_name: 'documents',
      llm: @llm
    )
    @vector_table ||= prepare_vector_table(erase:)
    @root_path = Pathname.new(Dir.pwd)
    load_sample_documents
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

    # Step 2: Use this hypothetical answer to find similar real documents with langchain similarity search
    # The embedding of the hypothetical answer helps find truly relevant content
    relevant_docs = vector_client.similarity_search_with_hyde(query: hypothetical_answer.completion, k: top_k)
    puts "\n2. Retrieved relevant documents using HyDE:"
    puts "--------------------------------"
    relevant_docs.each do |doc|
      puts "\nDocument id #{doc.id}:"
      puts doc.content
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

  # Generate the final answer based on the query and relevant documents
  # @param query [String] The original query
  # @param relevant_docs [Array<Hash>] An array of relevant documents
  # @return [Langchain::Completion] The final answer
  def generate_final_answer(query, relevant_docs)
    context = relevant_docs.map do |doc|
      "Source (document id #{doc.id}): #{doc.content}"
    end.join("\n\n")

    prompt = <<~PROMPT
      Question: #{query}

      Using only the information from these sources, provide a well-supported answer:
      #{context}

      Requirements:
      1. Only use information from the provided sources
      2. Cite the document id as source, when making specific claims
      3. If the sources don't fully answer the question, acknowledge this
      
      Answer:
    PROMPT

    llm.complete(prompt: prompt)
  end

  # Load sample documents for demonstration
  # @param erase [Boolean] Whether to erase existing documents
  # @return [Array<Hash>] An array of sample documents
  def load_sample_documents
    return unless vector_table.count.zero?

    documents_path = root_path.join("documents.json")
    documents = JSON.parse(File.read(documents_path))

    documents.each do |doc|
      load_document_to_vector_table(doc)
    end
  end

  # Load a document into the vector table based on its file path or content
  # @param doc [Hash] The document to load
  # @return [void]
  def load_document_to_vector_table(doc)
    if doc['file']
      file_path = root_path.join(doc['file'])
      vector_client.add_data(paths: [file_path])
    elsif doc['content']
      vector_client.add_texts(texts: [doc['content']])
    end
  end

  # Prepare the vector table for storing documents
  # @param erase [Boolean] Whether to erase existing documents
  # @return [Sequel::Dataset] The database table for storing documents
  def prepare_vector_table(erase:)
    db_table = vector_client.db.from(:documents)

    begin
      documents_count = db_table.count

      if documents_count > 0 && erase
        # Erase existing documents
        vector_client.destroy_default_schema
        vector_client.create_default_schema
      end
    rescue Sequel::DatabaseError => e
      # Create the table if it doesn't exist
      if e.message.include?('PG::UndefinedTable')
        vector_client.create_default_schema
      else
        raise
      end
    end

    db_table
  end
end

# Run the example
# This example demonstrates how to use the HyDE retriever to retrieve relevant documents for a given query
def demonstrate_hyde_rag
  # Get parameters from the command line
  query = ARGV[0] || "What effect does meditation have on the brain and stress levels?"
  erase = ARGV[1] == 'erase'

  puts "Initializing HyDE-RAG system..."
  hyde_retriever = HydeRetriever.new(erase:)

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