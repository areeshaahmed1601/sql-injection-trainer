import React, { useState } from "react";
import { submitChallenge } from "../utils/api";

const Challenges = () => {
  const [currentChallenge, setCurrentChallenge] = useState(0);
  const [userInput, setUserInput] = useState("");
  const [results, setResults] = useState({});
  const [showHint, setShowHint] = useState(false);
  const [loading, setLoading] = useState(false);

  const challenges = [
    {
      id: 1,
      title: "Login Bypass",
      difficulty: "Easy",
      points: 50,
      icon: "🔑",
      description: "Bypass the login authentication using SQL injection",
      scenario:
        "You need to login as admin without knowing the password. The application uses this query: SELECT * FROM users WHERE username = '[input]' AND password = '[password]'",
      hint: "Think about how you can make the WHERE clause always true using OR conditions",
      solution: "admin' OR '1'='1' --",
    },
    {
      id: 2,
      title: "Data Extraction",
      difficulty: "Medium",
      points: 75,
      icon: "📊",
      description: "Extract all user data using UNION injection",
      scenario:
        "The application shows user details based on ID. Extract all user data from the database",
      hint: "Use UNION to combine results from the users table with another query",
      solution: "1 UNION SELECT username, password FROM users",
    },
    {
      id: 3,
      title: "Database Schema",
      difficulty: "Hard",
      points: 100,
      icon: "🏗️",
      description: "Extract database schema information",
      scenario:
        "Find out what tables exist in the database to understand its structure",
      hint: "Information schema contains metadata about all tables in the database",
      solution: "1 UNION SELECT table_name FROM information_schema.tables",
    },
    {
      id: 4,
      title: "Blind Injection",
      difficulty: "Expert",
      points: 150,
      icon: "👁️",
      description:
        "Use time-based blind SQL injection to detect table existence",
      scenario:
        "Determine if an admin table exists without seeing direct results",
      hint: "Use time delays and conditional statements to detect table existence",
      solution: "1 AND (SELECT sleep(5) FROM admin WHERE 1=1)",
    },
  ];

  const handleSubmit = async () => {
    if (!userInput.trim()) return;

    setLoading(true);
    try {
      const response = await submitChallenge(
        challenges[currentChallenge].id,
        userInput
      );
      setResults((prev) => ({
        ...prev,
        [challenges[currentChallenge].id]: {
          ...response.data,
          userInput,
          timestamp: new Date().toLocaleTimeString(),
          completed: response.data.completed,
        },
      }));
    } catch (error) {
      setResults((prev) => ({
        ...prev,
        [challenges[currentChallenge].id]: {
          error: "Failed to check solution",
          timestamp: new Date().toLocaleTimeString(),
        },
      }));
    }
    setLoading(false);
  };

  const current = challenges[currentChallenge];
  const currentResult = results[current.id];

  return (
    <div className="max-w-6xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          SQL Injection Challenges
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Test your skills with real-world scenarios. Solve challenges to earn
          points and climb the leaderboard!
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Challenge Navigation */}
        <div className="lg:col-span-1">
          <div className="card p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Challenges</h2>
            <div className="space-y-3">
              {challenges.map((challenge, index) => (
                <button
                  key={challenge.id}
                  onClick={() => {
                    setCurrentChallenge(index);
                    setShowHint(false);
                    setUserInput("");
                  }}
                  className={`w-full text-left p-4 rounded-xl transition-all duration-200 ${
                    currentChallenge === index
                      ? "bg-gradient-to-r from-blue-50 to-purple-50 border-2 border-blue-200"
                      : "bg-gray-50 hover:bg-gray-100 border-2 border-transparent"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span className="text-xl">{challenge.icon}</span>
                      <div>
                        <h3 className="font-semibold text-gray-900">
                          {challenge.title}
                        </h3>
                        <p className="text-sm text-gray-500">
                          {challenge.difficulty}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-bold text-yellow-600">
                        {challenge.points} XP
                      </p>
                      {results[challenge.id]?.completed && (
                        <span className="text-green-600 text-sm">✅</span>
                      )}
                    </div>
                  </div>
                </button>
              ))}
            </div>

            {/* Progress Summary */}
            <div className="mt-6 p-4 bg-gray-50 rounded-xl">
              <h3 className="font-semibold text-gray-900 mb-3">
                Your Progress
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Completed:</span>
                  <span className="font-semibold">
                    {Object.values(results).filter((r) => r.completed).length}/
                    {challenges.length}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Total Points:</span>
                  <span className="font-semibold text-yellow-600">
                    {Object.values(results).reduce(
                      (total, r) => total + (r.score || 0),
                      0
                    )}{" "}
                    XP
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Current Challenge */}
        <div className="lg:col-span-3">
          <div className="card p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl flex items-center justify-center">
                  <span className="text-white text-2xl">{current.icon}</span>
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">
                    {current.title}
                  </h2>
                  <div className="flex items-center space-x-4">
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-medium ${
                        current.difficulty === "Easy"
                          ? "bg-green-100 text-green-800"
                          : current.difficulty === "Medium"
                          ? "bg-yellow-100 text-yellow-800"
                          : current.difficulty === "Hard"
                          ? "bg-orange-100 text-orange-800"
                          : "bg-red-100 text-red-800"
                      }`}
                    >
                      {current.difficulty}
                    </span>
                    <span className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm font-medium">
                      {current.points} XP
                    </span>
                  </div>
                </div>
              </div>
              {currentResult?.completed && (
                <span className="bg-green-100 text-green-800 px-4 py-2 rounded-full font-semibold">
                  ✅ Completed
                </span>
              )}
            </div>

            <p className="text-gray-700 text-lg mb-6">{current.description}</p>

            <div className="bg-blue-50 rounded-xl p-4 mb-6">
              <h3 className="font-semibold text-blue-900 mb-2">Scenario:</h3>
              <p className="text-blue-800">{current.scenario}</p>
            </div>

            {/* Input Area */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Your SQL Injection Payload:
                </label>
                <input
                  type="text"
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  placeholder="Enter your SQL injection payload here..."
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono"
                  disabled={currentResult?.completed}
                />
              </div>

              {/* Action Buttons */}
              <div className="flex space-x-4">
                <button
                  onClick={handleSubmit}
                  disabled={
                    loading || !userInput.trim() || currentResult?.completed
                  }
                  className="btn-primary flex-1 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                  {loading ? (
                    <div className="flex items-center justify-center space-x-2">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      <span>Checking...</span>
                    </div>
                  ) : (
                    "Submit Solution"
                  )}
                </button>
                <button
                  onClick={() => setShowHint(!showHint)}
                  className="btn-secondary"
                >
                  {showHint ? "Hide Hint" : "Show Hint"}
                </button>
              </div>

              {/* Hint */}
              {showHint && (
                <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
                  <h4 className="font-semibold text-yellow-800 mb-2">
                    💡 Hint:
                  </h4>
                  <p className="text-yellow-700">{current.hint}</p>
                </div>
              )}

              {/* Results */}
              {currentResult && (
                <div
                  className={`p-6 rounded-xl border-2 transition-all duration-300 ${
                    currentResult.completed
                      ? "bg-green-50 border-green-200"
                      : currentResult.error
                      ? "bg-red-50 border-red-200"
                      : "bg-orange-50 border-orange-200"
                  }`}
                >
                  <div className="flex items-center space-x-3 mb-4">
                    <div
                      className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                        currentResult.completed
                          ? "bg-green-100"
                          : currentResult.error
                          ? "bg-red-100"
                          : "bg-orange-100"
                      }`}
                    >
                      <span
                        className={`text-2xl ${
                          currentResult.completed
                            ? "text-green-600"
                            : currentResult.error
                            ? "text-red-600"
                            : "text-orange-600"
                        }`}
                      >
                        {currentResult.completed
                          ? "✅"
                          : currentResult.error
                          ? "❌"
                          : "⚠️"}
                      </span>
                    </div>
                    <div>
                      <h3
                        className={`text-xl font-bold ${
                          currentResult.completed
                            ? "text-green-800"
                            : currentResult.error
                            ? "text-red-800"
                            : "text-orange-800"
                        }`}
                      >
                        {currentResult.completed
                          ? "Challenge Completed!"
                          : currentResult.error
                          ? "Error"
                          : "Keep Trying!"}
                      </h3>
                      <p className="text-gray-600">
                        {currentResult.completed
                          ? `Earned ${currentResult.score} XP`
                          : currentResult.error
                          ? currentResult.error
                          : "Solution needs improvement"}
                      </p>
                    </div>
                  </div>

                  {currentResult.message && (
                    <p className="text-gray-700 mb-2">
                      {currentResult.message}
                    </p>
                  )}

                  {currentResult.detected_patterns &&
                    currentResult.detected_patterns.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-2">
                          Patterns Detected:
                        </h4>
                        <ul className="space-y-1">
                          {currentResult.detected_patterns.map(
                            (pattern, idx) => (
                              <li
                                key={idx}
                                className="flex items-center space-x-2 text-sm text-gray-700"
                              >
                                <span className="w-2 h-2 bg-red-400 rounded-full"></span>
                                <span>{pattern}</span>
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                    )}

                  <p className="text-sm text-gray-500 mt-4">
                    Submitted at: {currentResult.timestamp}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Challenges;
