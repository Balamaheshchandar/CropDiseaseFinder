// ==========================================
// 1. TRAINING DATASET
// ==========================================
const trainingData = {
  positive: [
    "beautiful movie with amazing story",
    "excellent acting and wonderful visuals",
    "heartwarming and enjoyable film",
    "great direction and fantastic performance"
  ],
  negative: [
    "boring movie with poor story",
    "terrible acting and bad visuals",
    "disappointing and slow film",
    "poor direction and boring performance"
  ]
};

// Common English stop words to filter out noise
const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
  "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
  "to", "was", "were", "will", "with"
]);

// ==========================================
// 2. TEXT PREPROCESSING
// ==========================================
function preprocess(text) {
  return text
    .toLowerCase()
    // Remove punctuation marks and symbols
    .replace(/[^a-z0-9\s]/g, " ")
    // Split into tokens by whitespace
    .split(/\s+/)
    // Remove stop words and empty tokens
    .filter(word => word.length > 0 && !STOP_WORDS.has(word));
}

// ==========================================
// 3. NAÏVE BAYES CLASSIFIER MODEL
// ==========================================
class NaiveBayesClassifier {
  constructor() {
    this.posWordCounts = {};
    this.negWordCounts = {};
    this.totalPosWords = 0;
    this.totalNegWords = 0;
    this.vocabulary = new Set();
    this.priorPos = 0.5;
    this.priorNeg = 0.5;
  }

  train(dataset) {
    const totalDocs = dataset.positive.length + dataset.negative.length;
    this.priorPos = dataset.positive.length / totalDocs;
    this.priorNeg = dataset.negative.length / totalDocs;

    // Process positive training reviews
    dataset.positive.forEach(review => {
      const words = preprocess(review);
      words.forEach(word => {
        this.posWordCounts[word] = (this.posWordCounts[word] || 0) + 1;
        this.totalPosWords++;
        this.vocabulary.add(word);
      });
    });

    // Process negative training reviews
    dataset.negative.forEach(review => {
      const words = preprocess(review);
      words.forEach(word => {
        this.negWordCounts[word] = (this.negWordCounts[word] || 0) + 1;
        this.totalNegWords++;
        this.vocabulary.add(word);
      });
    });
  }

  classify(text) {
    const words = preprocess(text);
    const vocabSize = this.vocabulary.size;

    // We use log-probabilities to prevent underflow from multiplying small floats:
    // log(P(Class | Review)) ∝ log(P(Class)) + Σ log(P(word_i | Class))
    let logPosScore = Math.log(this.priorPos);
    let logNegScore = Math.log(this.priorNeg);

    words.forEach(word => {
      // Laplace Smoothing (Add-1 smoothing):
      // P(word | Class) = (count(word, Class) + 1) / (totalWordsInClass + |Vocabulary|)
      const countInPos = this.posWordCounts[word] || 0;
      const countInNeg = this.negWordCounts[word] || 0;

      const probWordGivenPos = (countInPos + 1) / (this.totalPosWords + vocabSize);
      const probWordGivenNeg = (countInNeg + 1) / (this.totalNegWords + vocabSize);

      logPosScore += Math.log(probWordGivenPos);
      logNegScore += Math.log(probWordGivenNeg);
    });

    // Convert log-likelihoods back to normalized probability percentages:
    // Softmax-style stabilization to avoid Math.exp() overflow
    const maxLog = Math.max(logPosScore, logNegScore);
    const expPos = Math.exp(logPosScore - maxLog);
    const expNeg = Math.exp(logNegScore - maxLog);
    const sumExp = expPos + expNeg;

    const posProbability = (expPos / sumExp) * 100;
    const negProbability = (expNeg / sumExp) * 100;

    const prediction = posProbability >= negProbability ? "POSITIVE" : "NEGATIVE";

    return {
      prediction,
      posProbability: Math.round(posProbability),
      negProbability: Math.round(negProbability)
    };
  }
}

// Initialize and train model once loaded
const classifier = new NaiveBayesClassifier();
classifier.train(trainingData);

// ==========================================
// 4. UI INTERACTION & EVENT LISTENERS
// ==========================================
const reviewInput = document.getElementById("reviewInput");
const classifyBtn = document.getElementById("classifyBtn");
const exampleBtn = document.getElementById("exampleBtn");
const clearBtn = document.getElementById("clearBtn");
const resultSection = document.getElementById("resultSection");
const predictionBadge = document.getElementById("predictionBadge");
const posProb = document.getElementById("posProb");
const negProb = document.getElementById("negProb");
const explanationText = document.getElementById("explanationText");

const exampleReviewText = "The film was beautifully directed, with stunning visuals and a heartwarming story.";

// Classify button click handler
classifyBtn.addEventListener("click", () => {
  const text = reviewInput.value.trim();

  if (!text) {
    alert("Please enter a movie review first.");
    return;
  }

  // Run Naïve Bayes calculation
  const result = classifier.classify(text);

  // Update DOM elements
  predictionBadge.textContent = result.prediction;
  predictionBadge.className = `badge ${result.prediction === "POSITIVE" ? "badge-positive" : "badge-negative"}`;

  posProb.textContent = `${result.posProbability}%`;
  negProb.textContent = `${result.negProbability}%`;

  if (result.prediction === "POSITIVE") {
    explanationText.textContent = "The review contains words that are more strongly associated with positive reviews in the training dataset.";
  } else {
    explanationText.textContent = "The review contains words that are more strongly associated with negative reviews in the training dataset.";
  }

  resultSection.classList.remove("hidden");
});

// Load example button click handler
exampleBtn.addEventListener("click", () => {
  reviewInput.value = exampleReviewText;
});

// Clear button click handler
clearBtn.addEventListener("click", () => {
  reviewInput.value = "";
  resultSection.classList.add("hidden");
});