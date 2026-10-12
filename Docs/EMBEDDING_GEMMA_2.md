# EmbeddingGemma 2 text retrieval

`VecturaOAIKit.EmbeddingGemma2Embedder` integrates **text-only EmbeddingGemma 2** with a separately managed, local llama.cpp server. It adds no package dependencies and does not download weights, start processes, or accept model terms. This is a macOS local-server deployment path, not an in-process iOS/MLX runtime.

## Select weights and output dimensions

```swift
import Foundation
import VecturaKit
import VecturaOAIKit

let model = try EmbeddingGemma2Configuration(
  precision: .q8Zero,       // Or .bf16; select the matching server GGUF as well.
  dimensions: .d256,     // .d128, .d256, .d512, .d768
  batchSize: 8           // 1...32; batches are sent sequentially.
)
let embedder = try EmbeddingGemma2Embedder(
  configuration: model,
  baseURL: "http://127.0.0.1:8080/v1"
)
let database = try await VecturaKit(
  config: VecturaConfig(name: "gemma2-q8-256"),
  embedder: embedder
)
_ = try await database.addDocuments(texts: ["Actors isolate mutable state in Swift."])
let results = try await database.search(query: "How does Swift prevent data races?")
```

The pinned [ggml-org GGUF repository](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/tree/bfcd298762cc34d0357ece5ebdd31791a3a374d8) contains these text artifacts (verified 2026-10-12):

| `precision` | Artifact | Exact bytes | MiB |
| --- | --- | ---: | ---: |
| `.q8Zero` | `embeddinggemma-2-Q8_0.gguf` | 309,855,456 | 295.50 |
| `.bf16` | `embeddinggemma-2-BF16.gguf` | 557,950,176 | 532.10 |

Q8_0 reduces weight-file size by about 44% relative to BF16. Quantization can change retrieval quality; measure on your corpus. File size is **not** peak RAM: tokenizer, runtime, activation buffers, context and concurrent work also consume memory. BF16 avoids this weight quantization but takes more space. No Q4/Q5/Q6 or FP16 preset is advertised. The selected artifact URL, byte count and SHA256 are exposed on `Precision`; verify downloads against those values. Model weights are Apache 2.0; review the [model card and license](https://huggingface.co/google/embeddinggemma-2) before deployment. llama.cpp uses the [MIT license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE).

Output dimensions are a separate storage/quality choice: 128/256/512/768 Float values use 512/1024/2048/3072 raw bytes per vector. Persisted JSON and indexes have additional overhead. Truncation does not shrink model weights or avoid full 768-dimensional inference.

## Start the matching server

Use llama.cpp **b11474** (commit `b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea`) or a newer compatible release. Older runtimes that support the original 300M model do not necessarily support version 2. Supply a verified local GGUF from the pinned repository:

```sh
llama-server \
  --model /path/to/embeddinggemma-2-Q8_0.gguf \
  --alias embeddinggemma-2-Q8_0-bfcd298762cc34d0357ece5ebdd31791a3a374d8 \
  --embedding --pooling mean \
  --ctx-size 512 --batch-size 512 --ubatch-size 512 \
  --parallel 1 --threads 4 --n-gpu-layers 99 \
  --host 127.0.0.1 --port 8080 --no-webui
```

For BF16, change both the model filename and alias to BF16. `model.llamaServerArguments(modelDirectory:port:)` generates this argument array for a caller-owned `Process`. Selecting a precision in Swift selects the required server alias and launch artifact; it cannot re-quantize or replace already loaded weights. The provider checks `/v1/models` before every batch and rejects an absent alias. Operators must bind that alias to the specified, checksum-verified artifact; an alias alone cannot prove weight provenance.

This bounded context is intentional. The model supports up to 8,192 tokens, but server context/batch limits, prefixes and special tokens consume the actual input budget. Chunk documents to fit the server's tokenizer budget, or increase context and batch together after measuring memory. Over-limit/server/transport failures propagate. The Swift client does not approximate token counts or silently truncate strings.

## Retrieval contract and compatibility

The provider sends `title: none | text: {document}` for document embedding and `task: search result | query: {query}` for search. Pass **unformatted** text; prefixes are added once. The core `embedQuery(text:)` requirement defaults to the existing `embed(text:)` behavior for other providers, so their defaults stay intact. Both hybrid and vector search use the query hook; add/update use document embedding. Custom search engines must call the query hook themselves.

llama.cpp handles the GGUF tokenizer and special tokens, attention, mean pooling, and the learned projection. The client requires a native 768-vector, checks finite values, takes the requested leading dimensions, and L2-normalizes again. A zero truncated vector, malformed indices, vector count mismatch or wrong native dimension fails. See Google's [model card](https://huggingface.co/google/embeddinggemma-2) for prompting and Matryoshka guidance.

Default file storage writes `.vectura-embedding-space` when the embedder declares an identity. Changing model revision, precision, output dimension or prompt version requires a **new database and re-embedding**. Tagged stores reject incompatible or undeclared identities; nonempty untagged stores cannot be silently adopted. Reset retains the identity. Existing untagged providers/stores retain their behavior. Custom storage providers must enforce compatible metadata themselves; this guard covers default file storage and explicitly configured `FileStorageProvider` instances. Identity validation happens at initialization. Sharing a directory between simultaneously open database/provider instances (within one process or across processes) is unsupported: close existing instances before changing providers, including an untagged instance opened while the directory was empty.

The original [EmbeddingGemma 300M](https://huggingface.co/google/embeddinggemma-300m) is a separate Gemma 3 text model with a 2K context and Gemma license; version 2 uses Gemma 4, a 270M text component and optional multimodal encoders. Their weights, tokenizers and embedding spaces are not interchangeable despite sharing retrieval-prefix syntax and 768 output dimensions. This provider **does not claim original 300M, MLX, image, audio or video support**. VecturaKit's MLX backend lives in [VecturaMLXKit](https://github.com/rryam/VecturaMLXKit).

## Test gates

```sh
# Unit/HTTP fixture and core regression tests; no weights or server needed.
swift test --no-parallel --skip PerformanceTests

# Explicit real inference gate: start the matching server first, then:
VECTURA_GEMMA2_INTEGRATION=1 \
VECTURA_GEMMA2_BASE_URL=http://127.0.0.1:8080/v1 \
VECTURA_GEMMA2_PRECISION=Q8_0 \
swift test --no-parallel --filter EmbeddingGemma2IntegrationTests
```

Use `VECTURA_GEMMA2_PRECISION=BF16` with the BF16 server to validate that artifact. This opt-in test checks real batched/single inference, all four dimensions, unit norms, semantic retrieval and persisted database reopening. The ordinary unit suite uses HTTP fixtures and does not certify inference or retrieval quality. CI runs the unit suite; `RUN_GEMMA2_INTEGRATION=1` enables the optional gate only when a compatible server has been provisioned externally. No test downloads a model.

Sources: [Google documentation](https://ai.google.dev/gemma/docs/embeddinggemma), [launch post](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/), [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea/tools/server).
