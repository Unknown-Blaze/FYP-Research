import argparse
from rapidfuzz import distance as levenshtein_distance
from Bio.Align import PairwiseAligner
from Bio import Align

def calculate_levenshtein_similarity(seq1, seq2):
    """
    Calculates similarity based on Levenshtein distance.
    The score is normalized to a 0-100 percentage.
    """
    seq1, seq2 = str(seq1), str(seq2)
    if not seq1 and not seq2:
        return 100.0
    if not seq1 or not seq2:
        return 0.0

    # Calculate the Levenshtein distance (number of edits)
    dist = levenshtein_distance.Levenshtein.distance(seq1, seq2)
    
    # Normalize to a similarity score
    longer_len = max(len(seq1), len(seq2))
    similarity = (1 - (dist / longer_len)) * 100
    return similarity

def calculate_alignment_similarity(seq1, seq2):
    """
    Calculates similarity based on a standard Needleman-Wunsch global alignment
    using the BLOSUM62 scoring matrix. The score is normalized to a 0-100 percentage.
    """
    seq1, seq2 = str(seq1), str(seq2)
    if not seq1 and not seq2:
        return 100.0
    if not seq1 or not seq2:
        return 0.0

    # Configure the aligner
    aligner = PairwiseAligner()
    aligner.mode = 'global'  # Use Needleman-Wunsch for global alignment
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    
    # Use a standard biological scoring matrix
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")

    # Perform the alignment and get the score
    # The score is the sum of scores for matches, mismatches, and gaps.
    alignment_score = aligner.score(seq1, seq2)

    # To normalize, we calculate the theoretical maximum score (aligning a sequence to itself)
    # and the theoretical minimum (aligning to a sequence of gaps). This provides a robust 0-100 scale.
    score_seq1_self = aligner.score(seq1, seq1)
    score_seq2_self = aligner.score(seq2, seq2)
    
    # Normalize the score to a percentage
    # We normalize against the smaller of the two self-alignment scores to be conservative
    normalized_score = (alignment_score / min(score_seq1_self, score_seq2_self)) * 100
    
    # Ensure the score is between 0 and 100
    return max(0, normalized_score)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the similarity between two sequences using different methods.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("seq1", help="The first sequence.")
    parser.add_argument("seq2", help="The second sequence.")
    parser.add_argument(
        "--method", 
        choices=['levenshtein', 'alignment'], 
        default='alignment',
        help="""The method to use for comparison:
  - levenshtein: Fast, simple edit distance. Good for general strings.
  - alignment:   Standard bioinformatics global alignment with BLOSUM62. 
                 **Recommended for protein sequences.**
"""
    )
    args = parser.parse_args()

    if args.method == 'levenshtein':
        similarity = calculate_levenshtein_similarity(args.seq1, args.seq2)
    else: # alignment
        similarity = calculate_alignment_similarity(args.seq1, args.seq2)

    print(f"Method: {args.method}")
    print(f"Sequence 1: {args.seq1}")
    print(f"Sequence 2: {args.seq2}")
    print(f"Similarity: {similarity:.2f}%")


if __name__ == "__main__":
    main()