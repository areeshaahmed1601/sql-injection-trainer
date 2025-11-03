import React, { useState, useEffect } from "react";
import {
  checkSQLInjection,
  getDetectionStats,
  getModelInfo,
} from "../utils/api";

const Training = () => {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeModule, setActiveModule] = useState(0);
  const [stats, setStats] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  const trainingModules = [
    {
      title: "Basic SQL Injection",
      level: "Beginner",
      icon: "🔍",
      description:
        "Learn about basic SQL injection techniques using tautologies",
      example:
        "SELECT * FROM users WHERE username = 'admin' OR '1'='1' --' AND password = 'any'",
      explanation:
        "This query bypasses authentication because '1'='1' is always true, making the WHERE clause always return results.",
      patterns: [
        "OR statements",
        "Always true conditions",
        "Comment operators",
      ],
    },
    {
      title: "Union-Based Injection",
      level: "Intermediate",
      icon: "🔄",
      description:
        "Understand how UNION queries can extract data from other tables",
      example:
        "SELECT name, email FROM users WHERE id=1 UNION SELECT username, password FROM admin",
      explanation:
        "UNION combines results from multiple SELECT statements, allowing attackers to extract data from different tables.",
      patterns: [
        "UNION operator",
        "Multiple SELECT statements",
        "Column matching",
      ],
    },
    {
      title: "Time-Based Blind Injection",
      level: "Advanced",
      icon: "⏱️",
      description: "Learn about time-based detection techniques",
      example: "SELECT * FROM users WHERE id=1 AND (SELECT sleep(5))",
      explanation:
        "This causes the database to pause, revealing information through timing differences in response.",
      patterns: ["SLEEP functions", "Time delays", "Conditional responses"],
    },
    {
      title: "Error-Based Injection",
      level: "Expert",
      icon: "🚨",
      description: "Extract information through database error messages",
      example:
        "SELECT * FROM users WHERE id=1 AND updatexml(1, concat(0x7e, (SELECT version())), 1)",
      explanation:
        "Forces the database to generate error messages containing sensitive information.",
      patterns: [
        "Error generation",
        "Information leakage",
        "Database functions",
      ],
    },
  ];

  useEffect(() => {
    loadStats();
    loadModelInfo();
  }, []);

  const loadStats = async () => {
    try {
      const response = await getDetectionStats();
      setStats(response.data);
    } catch (error) {
      console.error("Failed to load stats:", error);
    }
  };

  const loadModelInfo = async () => {
    try {
      const response = await getModelInfo();
      setModelInfo(response.data);
    } catch (error) {
      console.error("Failed to load model info:", error);
    }
  };

  const handleCheckQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await checkSQLInjection(query);
      setResult(response.data);
      // Refresh stats to show new detection
      loadStats();
    } catch (error) {
      setResult({
        is_malicious: false,
        confidence: 0,
        message: "Error checking query: " + error.message,
        detected_patterns: [],
        detection_method: "error",
        risk_level: "low",
      });
    }
    setLoading(false);
  };

  const currentModule = trainingModules[activeModule];

  return (
    <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          SQL Injection Training
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Master SQL injection detection through comprehensive training modules
          with our 99.4% accurate ML model.
        </p>

        {modelInfo && (
          <div className="mt-4 inline-flex items-center px-4 py-2 bg-green-100 text-green-800 rounded-full text-sm font-medium">
            🎯 ML Model: {modelInfo.status} | Accuracy: 99.4%
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Stats Sidebar */}
        <div className="lg:col-span-1 space-y-6">
          {stats && (
            <>
              <div className="card p-6">
                <h3 className="text-lg font-bold mb-4">Detection Analytics</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span>Total Queries:</span>
                    <span className="font-bold">
                      {stats.detection_stats?.total_queries || 0}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Malicious Detected:</span>
                    <span className="font-bold text-red-600">
                      {stats.detection_stats?.malicious_count || 0}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Average Confidence:</span>
                    <span className="font-bold text-blue-600">
                      {stats.detection_stats?.average_confidence || 0}%
                    </span>
                  </div>
                </div>
              </div>

              {stats.common_patterns && stats.common_patterns.length > 0 && (
                <div className="card p-6">
                  <h3 className="text-lg font-bold mb-4">Common Patterns</h3>
                  <div className="space-y-2">
                    {stats.common_patterns.slice(0, 5).map((pattern, idx) => (
                      <div key={idx} className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                        <span className="text-sm">{pattern}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3 space-y-8">
          {/* Module Navigation */}
          <div className="card p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              Training Modules
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {trainingModules.map((module, index) => (
                <button
                  key={index}
                  onClick={() => setActiveModule(index)}
                  className={`text-left p-4 rounded-xl transition-all duration-200 ${
                    activeModule === index
                      ? "bg-gradient-to-r from-blue-50 to-purple-50 border-2 border-blue-200"
                      : "bg-gray-50 hover:bg-gray-100 border-2 border-transparent"
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{module.icon}</span>
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900">
                        {module.title}
                      </h3>
                      <p className="text-sm text-gray-500">{module.level}</p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Current Module Content */}
          <div className="card p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl flex items-center justify-center">
                  <span className="text-white text-2xl">
                    {currentModule.icon}
                  </span>
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">
                    {currentModule.title}
                  </h2>
                  <p className="text-gray-500">{currentModule.level} Level</p>
                </div>
              </div>
              <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                25 XP Available
              </span>
            </div>

            <p className="text-gray-700 mb-6 text-lg">
              {currentModule.description}
            </p>

            <div className="bg-gray-50 rounded-xl p-4 mb-6">
              <h3 className="font-semibold text-gray-900 mb-3">
                Example Attack:
              </h3>
              <code className="bg-gray-800 text-green-400 p-4 rounded-lg block text-sm font-mono overflow-x-auto">
                {currentModule.example}
              </code>
            </div>

            <div className="bg-blue-50 rounded-xl p-4 mb-6">
              <h3 className="font-semibold text-blue-900 mb-2">
                How it works:
              </h3>
              <p className="text-blue-800">{currentModule.explanation}</p>
            </div>

            <div>
              <h3 className="font-semibold text-gray-900 mb-3">
                Patterns to Detect:
              </h3>
              <div className="flex flex-wrap gap-2">
                {currentModule.patterns.map((pattern, idx) => (
                  <span
                    key={idx}
                    className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm"
                  >
                    {pattern}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Practice Area */}
          <div className="card p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              Practice Detection
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Test Your Knowledge - Enter a SQL Query:
                </label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Paste or type a SQL query here to check for injection patterns..."
                  className="w-full h-32 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none font-mono text-sm"
                />
              </div>

              <button
                onClick={handleCheckQuery}
                disabled={loading || !query.trim()}
                className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Analyzing with ML...</span>
                  </div>
                ) : (
                  "🔍 Analyze for SQL Injection"
                )}
              </button>

              {result && (
                <div
                  className={`p-6 rounded-xl border-2 transition-all duration-300 ${
                    result.is_malicious
                      ? "bg-red-50 border-red-200"
                      : "bg-green-50 border-green-200"
                  }`}
                >
                  <div className="flex items-center space-x-3 mb-4">
                    <div
                      className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                        result.is_malicious ? "bg-red-100" : "bg-green-100"
                      }`}
                    >
                      <span
                        className={`text-2xl ${
                          result.is_malicious
                            ? "text-red-600"
                            : "text-green-600"
                        }`}
                      >
                        {result.is_malicious ? "⚠️" : "✅"}
                      </span>
                    </div>
                    <div>
                      <h3
                        className={`text-xl font-bold ${
                          result.is_malicious
                            ? "text-red-800"
                            : "text-green-800"
                        }`}
                      >
                        {result.is_malicious
                          ? "Malicious Query Detected!"
                          : "Safe Query"}
                      </h3>
                      <p className="text-gray-600">
                        Confidence: {result.confidence}% | Method:{" "}
                        {result.detection_method}
                      </p>
                    </div>
                  </div>

                  <p className="text-gray-700 mb-4">{result.message}</p>

                  {result.detected_patterns &&
                    result.detected_patterns.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-2">
                          Detected Patterns:
                        </h4>
                        <ul className="space-y-1">
                          {result.detected_patterns.map((pattern, idx) => (
                            <li
                              key={idx}
                              className="flex items-center space-x-2 text-sm text-gray-700"
                            >
                              <span className="w-2 h-2 bg-red-400 rounded-full"></span>
                              <span>{pattern}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Training;
