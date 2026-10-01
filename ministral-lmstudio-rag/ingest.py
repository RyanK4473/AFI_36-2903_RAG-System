"""Build a local vector index from PDF, TXT, and Markdown documents."""
import json

from config import (
    DOCUMENTS_DIR, INDEX_PATH, EMBEDDING_MODEL,
    CHUNK_TOKENS, OVERLAP_TOKENS,
)


def read_documents():
    from pypdf import PdfReader

    for path in sorted(DOCUMENTS_DIR.rglob("*")):
        if not path.is_file():
            continue
        source = path.relative_to(DOCUMENTS_DIR).as_posix()
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(path)
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    yield source, page_number, text
                else:
                    print(f"No text extracted: {source}, PDF page {page_number}")
        elif suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8-sig")
            if text.strip():
                yield source, None, text


def split_text(text, tokenizer, chunk_tokens=CHUNK_TOKENS,
               overlap_tokens=OVERLAP_TOKENS):
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("Overlap must be smaller than the chunk size.")
    encoded = tokenizer(
        text, add_special_tokens=False, truncation=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    for start in range(0, len(offsets), chunk_tokens - overlap_tokens):
        end = min(start + chunk_tokens, len(offsets))
        # Slice the original text so case, punctuation, and section numbers
        # survive even though the embedding tokenizer is case-insensitive.
        chunk = text[offsets[start][0]:offsets[end - 1][1]].strip()
        if chunk:
            yield chunk
        if end == len(offsets):
            break


def main():
    import numpy as np
    from sentence_transformers import SentenceTransformer

    DOCUMENTS_DIR.mkdir(exist_ok=True)
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    chunks = []
    for source, page_number, text in read_documents():
        for chunk in split_text(text, embedder.tokenizer):
            length = len(embedder.tokenizer(chunk, truncation=False)["input_ids"])
            if length > embedder.max_seq_length:
                raise ValueError(
                    f"Chunk too long in {source}. Reduce CHUNK_TOKENS in config.py."
                )
            chunks.append({"source": source, "page": page_number, "text": chunk})
    if not chunks:
        raise SystemExit("No text found. Add PDF/TXT/MD files to documents; scanned PDFs need OCR.")

    vectors = embedder.encode(
        [chunk["text"] for chunk in chunks], batch_size=32,
        normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    metadata = {"format_version": 1, "embedding_model": EMBEDDING_MODEL,
                "chunks": chunks}
    temporary = INDEX_PATH.with_name("index.building.npz")
    np.savez_compressed(temporary, vectors=vectors,
                        metadata=json.dumps(metadata, ensure_ascii=False))
    temporary.replace(INDEX_PATH)
    print(f"Indexed {len(chunks)} chunks from {len(set(c['source'] for c in chunks))} files.")
    print(f"Saved: {INDEX_PATH}")


if __name__ == "__main__":
    main()
