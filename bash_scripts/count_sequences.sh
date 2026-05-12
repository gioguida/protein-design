#!/bin/bash
# Counts sequences in every dataset file under DATA_DIR and appends new rows
# to sequence_counts.csv. Re-runs skip already-counted files.

DATA_DIR="/cluster/project/infk/krause/mdenegri/protein-design/data"
OUTPUT_CSV="$DATA_DIR/sequence_counts.csv"

if [[ ! -f "$OUTPUT_CSV" ]]; then
    echo "file,n_sequences" > "$OUTPUT_CSV"
fi

# Build set of already-processed relative paths
declare -A processed
while IFS=, read -r file _rest; do
    processed["$file"]=1
done < <(tail -n +2 "$OUTPUT_CSV")

count_sequences() {
    local filepath="$1"
    if [[ "$filepath" == *.csv.gz ]]; then
        echo $(( $(zcat "$filepath" | wc -l) - 1 ))
    elif [[ "$filepath" == *.fasta ]]; then
        grep -c "^>" "$filepath"
    elif [[ "$filepath" == *.csv ]]; then
        echo $(( $(wc -l < "$filepath") - 1 ))
    fi
}

while IFS= read -r filepath; do
    relpath="${filepath#"$DATA_DIR/"}"

    if [[ -n "${processed[$relpath]}" ]]; then
        echo "Skipping (already counted): $relpath"
        continue
    fi

    echo "Counting: $relpath"
    n=$(count_sequences "$filepath")
    echo "$relpath,$n" >> "$OUTPUT_CSV"
done < <(find "$DATA_DIR" \
    -path "*/_backup*" -prune -o \
    -name "sequence_counts.csv" -prune -o \
    \( -name "*.fasta" -o -name "*.csv" -o -name "*.csv.gz" \) -print \
    | sort)

echo "Done. Results saved to $OUTPUT_CSV"
