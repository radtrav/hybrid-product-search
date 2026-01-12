# PRD: BM25 Implementation for Reranker

## Feature Overview
Replace the simple term overlap text matching algorithm with BM25 (Best Matching 25) to improve search ranking quality.

## Status: COMPLETE ✅

### Progress Tracker
- [x] Phase 1 (RED): Write all tests for BM25 components
  - [x] Created test_bm25.py with 20 unit tests
  - [x] Added integration tests to test_rerank.py
  - [x] Confirmed RED state (all tests failing as expected)
- [x] Phase 2 (GREEN): Implement BM25 to pass all tests
  - [x] Updated feature_extractor.py with BM25 methods
  - [x] Updated reranker.py with score normalization
  - [x] Run tests to confirm GREEN state (26/26 tests passing)
- [x] Phase 3 (REFACTOR): Clean up and optimize code
  - [x] Code review completed
  - [x] Docstrings added
  - [x] Backward compatibility maintained
- [x] Phase 4: End-to-end verification
  - [x] All 26 BM25 tests passing
  - [x] Implementation summary documented
  - [x] Ranking quality verified through tests

## Requirements

### Functional Requirements
1. Implement custom BM25 algorithm (no external libraries)
2. Use standard BM25 parameters: k1=1.5, b=0.75
3. Replace existing text_match feature (not adding as separate feature)
4. Maintain backward compatibility (fallback when all_candidates not provided)
5. Normalize BM25 scores to [0, 1] range using min-max normalization

### Technical Requirements
1. Pass all candidates to feature extractor for corpus statistics
2. Compute IDF (Inverse Document Frequency) from candidate batch
3. Apply length normalization using parameter b=0.75
4. Combine title and description as single document
5. Simple tokenization (lowercase + whitespace split)
6. No stopword removal or stemming in initial implementation

## Design Decisions

### 1. Document Representation
- **Decision**: Combine title and description into single document
- **Rationale**: Simpler implementation, matches current behavior

### 2. Corpus Statistics
- **Decision**: Use batch-level statistics (compute from current candidate set)
- **Rationale**: Minimal architectural changes, appropriate for reranking use case

### 3. Score Normalization
- **Decision**: Min-max normalization to [0,1]
- **Rationale**: Preserves relative ranking, consistent with other features

### 4. Text Preprocessing
- **Decision**: Lowercase + whitespace splitting only
- **Rationale**: Keep initial implementation simple, can enhance later

## Implementation Details

### Modified Files
1. `reranker/services/feature_extractor.py`
   - Added _tokenize() method
   - Added _compute_term_frequency() method
   - Added _compute_idf() method
   - Added _compute_bm25() method
   - Updated extract_features() signature with all_candidates parameter
   - Kept _compute_text_match() for backward compatibility

2. `reranker/services/reranker.py`
   - Added _normalize_scores() method
   - Modified rerank() to use two-pass approach (extract features, normalize BM25, score)

### New Test Files
1. `tests/test_bm25.py` - 20 unit tests covering:
   - Tokenization (3 tests)
   - Term frequency (3 tests)
   - IDF calculation (3 tests)
   - BM25 scoring (6 tests)
   - Score normalization (4 tests)
   - Backward compatibility (1 test)

2. `tests/test_rerank.py` - Added 4 integration tests:
   - BM25 reranking behavior
   - Rare term weighting
   - Length normalization
   - Backward compatibility

## Expected Improvements
1. **Term weighting**: Rare query terms weighted more heavily (via IDF)
2. **Frequency saturation**: Better handling of repeated terms (diminishing returns)
3. **Length normalization**: Prevents bias toward longer documents
4. **Industry standard**: Widely used in search engines

## Future Enhancements (Not in Scope)
1. Title boosting (multiply title term frequencies)
2. Stemming/lemmatization for better recall
3. N-gram support for phrase matching
4. BM25+ variant for edge case handling
5. Cached global statistics from full database

## Testing Strategy
- Test-Driven Development (TDD) with Red-Green-Refactor
- 20+ unit tests for BM25 components
- Integration tests for end-to-end behavior
- Backward compatibility tests

## Success Criteria
- [x] All unit tests pass (20/20)
- [x] All integration tests pass (4/4)
- [x] Existing tests continue to pass (2/2)
- [x] API endpoints work correctly (backward compatible)
- [x] Scores are in [0,1] range (verified)
- [x] BM25 ranks better than simple overlap for test queries (verified)

## Plan Reference
Full implementation plan: `/Users/T959741/.claude/plans/streamed-sniffing-moore.md`
