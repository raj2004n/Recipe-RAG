"""The two passes that turn the teacher's answers into the rows a trainer reads.

A training set is a draw and a render. The draw sorts the answer rows into train,
dev and eval out of the pools the category sidecar and the pair structure
describe, checks the counts it was drawn to, and records the split with the leak
it was measured against. The render walks the split, puts each row through the
student's own prompt path, and writes the masked context and the supervised
answer a trainer reads back, beside the manifest that names the inputs it read.

Each pass is reached through its own module, and the shared pieces sit beside
them: the paths and the numbers both read, the loaders that read the files a pass
works from, and the writes each one leaves behind. Nothing is re-exported here,
so that a caller who wants only the draw does not pull the tokenizer and torch in
with it, and the constants stay reachable on their own.
"""
