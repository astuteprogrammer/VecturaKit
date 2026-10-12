import Foundation
import Testing
import VecturaKit
import VecturaOAIKit

@Suite("EmbeddingGemma2 real llama.cpp inference", .serialized,
  .enabled(if: ProcessInfo.processInfo.environment["VECTURA_GEMMA2_INTEGRATION"] == "1"))
struct EmbeddingGemma2IntegrationTests {
  @Test("Real text inference, Matryoshka vectors, retrieval and database reopen")
  func realInference() async throws {
    let environment = ProcessInfo.processInfo.environment
    let url = environment["VECTURA_GEMMA2_BASE_URL"] ?? "http://127.0.0.1:8080/v1"
    let precision = try #require(EmbeddingGemma2Configuration.Precision(
      rawValue: environment["VECTURA_GEMMA2_PRECISION"] ?? "Q8_0"
    ))
    let documents = [
      "The northern lights are caused by charged particles from the Sun interacting with Earth's atmosphere.",
      "Sourdough bread is made by fermenting flour and water with a starter culture.",
      "The swift programming language uses actors to coordinate access to mutable state."
    ]
    for dimensions in EmbeddingGemma2Configuration.Dimensions.allCases {
      let config = try EmbeddingGemma2Configuration(precision: precision, dimensions: dimensions, batchSize: 2)
      let embedder = try EmbeddingGemma2Embedder(configuration: config, baseURL: url)
      let batch = try await embedder.embed(texts: documents)
      let single = try await embedder.embed(text: documents[0])
      #expect(batch.count == documents.count)
      #expect(zip(batch[0], single).allSatisfy { abs($0 - $1) < 0.002 })
      for vector in batch {
        #expect(vector.count == dimensions.rawValue)
        #expect(vector.allSatisfy { $0.isFinite })
        #expect(abs(vector.reduce(0) { $0 + $1 * $1 } - 1) < 0.0001)
      }
      let directory = URL(filePath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: directory) }
      let databaseConfig = try VecturaConfig(
        name: "gemma2", directoryURL: directory,
        searchOptions: .init(minThreshold: nil, hybridWeight: 1)
      )
      let database = try await VecturaKit(config: databaseConfig, embedder: embedder)
      let identifiers = try await database.addDocuments(texts: documents)
      let results = try await database.search(query: "What causes the aurora?", numResults: 3)
      #expect(results.first?.id == identifiers[0])
      let reopened = try await VecturaKit(config: databaseConfig, embedder: embedder)
      #expect(try await reopened.documentCount == 3)
      let persisted = try await reopened.search(query: "How do I ferment bread?", numResults: 3)
      #expect(persisted.first?.id == identifiers[1])
    }
  }
}
