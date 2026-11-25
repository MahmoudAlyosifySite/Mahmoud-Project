package com.mahmoudalyosify.auditor.core

/** Confusion matrix that grows as new labels appear (used by the lab screen and tests). */
class ConfusionMatrix(labels: List<String> = emptyList()) {
    private val labelList: MutableList<String> = labels.toMutableList()
    private val countRows: MutableList<MutableList<Int>> = MutableList(labels.size) { MutableList(labels.size) { 0 } }

    val labels: List<String> get() = labelList.toList()
    val counts: List<List<Int>> get() = countRows.map { it.toList() }

    fun add(truth: String, predicted: String) {
        val t = indexAdding(truth)
        val p = indexAdding(predicted)
        countRows[t][p] = countRows[t][p] + 1
    }

    val accuracy: Double
        get() {
            var total = 0
            var correct = 0
            for ((i, row) in countRows.withIndex()) {
                for ((j, c) in row.withIndex()) {
                    total += c
                    if (i == j) correct += c
                }
            }
            return if (total > 0) correct.toDouble() / total.toDouble() else 0.0
        }

    val total: Int get() = countRows.sumOf { it.sum() }

    fun count(truth: String, predicted: String): Int {
        val t = labelList.indexOf(truth)
        val p = labelList.indexOf(predicted)
        if (t < 0 || p < 0 || t >= countRows.size || p >= countRows[t].size) return 0
        return countRows[t][p]
    }

    private fun indexAdding(label: String): Int {
        if (!labelList.contains(label)) labelList.add(label)
        val size = labelList.size
        while (countRows.size > size) countRows.removeAt(countRows.size - 1)
        while (countRows.size < size) countRows.add(ArrayList())
        for (row in countRows) {
            while (row.size > size) row.removeAt(row.size - 1)
            while (row.size < size) row.add(0)
        }
        return labelList.indexOf(label)
    }
}
