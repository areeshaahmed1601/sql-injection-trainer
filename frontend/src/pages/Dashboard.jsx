import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  getDetectionStats,
  getUserProgress,
  getModelInfo,
  checkSQLInjection,
  checksqlInjection,
} from "../utils/api";

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [userProgress, setUserProgress] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  // State for SQL query checking
  const [queryInput, setQueryInput] = useState("");
  const [checkingQuery, setCheckingQuery] = useState(false);
  const [queryResult, setQueryResult] = useState(null);
  const [showQueryChecker, setShowQueryChecker] = useState(false);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      const [statsResponse, progressResponse, modelResponse] =
        await Promise.all([
          getDetectionStats(),
          getUserProgress(1),
          getModelInfo(),
        ]);

      setStats(statsResponse.data);
      setUserProgress(progressResponse.data);
      setModelInfo(modelResponse.data);
    } catch (error) {
      console.error("Failed to load dashboard data:", error);
    } finally {
      setLoading(false);
    }
  };

  // Function to check SQL query
  const checkQuery = async () => {
    if (!queryInput.trim()) {
      alert("Please enter a SQL query to check");
      return;
    }

    setCheckingQuery(true);
    setQueryResult(null);

    try {
      const response = await checkSQLInjection(queryInput);
      setQueryResult(response.data);

      // Refresh stats to update query count
      const statsResponse = await getDetectionStats();
      setStats(statsResponse.data);
    } catch (error) {
      console.error("Error checking query:", error);
      setQueryResult({
        is_malicious: false,
        confidence: 0,
        message: "Error analyzing query. Please try again.",
        detection_method: "error",
      });
    } finally {
      setCheckingQuery(false);
    }
  };

  // Function to try example queries
  const tryExampleQuery = (example) => {
    setQueryInput(example);
    setShowQueryChecker(true);
  };

  // Example queries for quick testing
  const exampleQueries = [
    {
      query: "SELECT * FROM users WHERE id = 1",
      description: "Normal SELECT query",
      type: "benign",
    },
    {
      query: "admin' OR '1'='1'",
      description: "Tautology attack",
      type: "malicious",
    },
    {
      query: "1 UNION SELECT username, password FROM users",
      description: "Union-based attack",
      type: "malicious",
    },
    {
      query: "UPDATE users SET last_login = NOW() WHERE id = 1",
      description: "Normal UPDATE",
      type: "benign",
    },
    {
      query: "1; DROP TABLE users --",
      description: "Stacked query attack",
      type: "malicious",
    },
    {
      query: "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
      description: "Normal COUNT query",
      type: "benign",
    },
  ];

  const getLevel = (score) => {
    if (score >= 800) return "Expert";
    if (score >= 500) return "Advanced";
    if (score >= 200) return "Intermediate";
    return "Beginner";
  };

  const calculateLevelProgress = (score) => {
    if (score >= 800) return 100;
    if (score >= 500) return 75;
    if (score >= 200) return 50;
    return (score / 200) * 50;
  };

  const dashboardStats = [
    {
      label: "Training Completed",
      value: userProgress
        ? `${userProgress.user_info.challenges_completed}/4`
        : "0/4",
      progress: userProgress
        ? (userProgress.user_info.challenges_completed / 4) * 100
        : 0,
      color: "from-green-500 to-emerald-500",
      icon: "✅",
    },
    {
      label: "Challenges Solved",
      value: userProgress
        ? userProgress.user_info.challenges_completed.toString()
        : "0",
      progress: userProgress
        ? (userProgress.user_info.challenges_completed / 4) * 100
        : 0,
      color: "from-blue-500 to-cyan-500",
      icon: "⚡",
    },
    {
      label: "Current Level",
      value: userProgress
        ? getLevel(userProgress.user_info.total_score)
        : "Beginner",
      progress: userProgress
        ? calculateLevelProgress(userProgress.user_info.total_score)
        : 0,
      color: "from-purple-500 to-pink-500",
      icon: "📊",
    },
    {
      label: "Total Points",
      value: userProgress ? userProgress.user_info.total_score.toString() : "0",
      progress: userProgress
        ? (userProgress.user_info.total_score / 1000) * 100
        : 0,
      color: "from-orange-500 to-red-500",
      icon: "⭐",
    },
  ];

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-gray-600">Loading dashboard data...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
      {/* Welcome Section */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          Welcome to SQL Shield! 🛡️
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl">
          Master SQL injection detection through interactive training and
          real-world challenges. Protect your applications while earning points
          and climbing the leaderboard.
        </p>

        {/* SQL Query Checker Section */}
        <div className="mt-8 bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-xl border border-blue-200 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
              <span className="bg-blue-100 p-2 rounded-lg">🔍</span>
              Real-time SQL Query Analyzer
            </h2>
            <button
              onClick={() => setShowQueryChecker(!showQueryChecker)}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              {showQueryChecker ? (
                <>
                  <span>👁️</span> Hide Analyzer
                </>
              ) : (
                <>
                  <span>🚀</span> Try It Now
                </>
              )}
            </button>
          </div>

          {showQueryChecker && (
            <>
              <div className="mb-6">
                <p className="text-gray-700 mb-4 text-lg">
                  <span className="font-semibold">
                    Try our ML-powered detection:
                  </span>{" "}
                  Enter any SQL query below to see real-time analysis using our
                  22-feature Random Forest model.
                </p>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Input Section */}
                  <div className="lg:col-span-2">
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        SQL Query Input
                      </label>
                      <textarea
                        value={queryInput}
                        onChange={(e) => setQueryInput(e.target.value)}
                        placeholder="Enter SQL query here... Example: SELECT * FROM users WHERE id=1 OR 1=1"
                        className="w-full h-40 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm bg-white shadow-sm"
                        rows={4}
                      />
                    </div>

                    <div className="flex gap-3">
                      <button
                        onClick={checkQuery}
                        disabled={checkingQuery || !queryInput.trim()}
                        className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-6 py-3 rounded-lg font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md"
                      >
                        {checkingQuery ? (
                          <>
                            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                            Analyzing with ML...
                          </>
                        ) : (
                          <>
                            <span className="text-lg">🔍</span>
                            Analyze Query
                          </>
                        )}
                      </button>

                      <button
                        onClick={() => setQueryInput("")}
                        className="px-6 py-3 border border-gray-300 text-gray-700 hover:bg-gray-50 rounded-lg font-medium transition-colors"
                      >
                        Clear
                      </button>
                    </div>

                    <div className="mt-4 text-sm text-gray-500">
                      <p className="flex items-center gap-2">
                        <span className="text-blue-600">💡</span>
                        Detection takes approximately{" "}
                        <span className="font-semibold">5ms</span> using our
                        Random Forest model
                      </p>
                    </div>
                  </div>

                  {/* Result Display */}
                  <div className="lg:col-span-1">
                    <div className="sticky top-4">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">
                        Analysis Results
                      </h3>

                      {queryResult ? (
                        <div
                          className={`p-5 rounded-xl border-2 shadow-sm ${
                            queryResult.is_malicious
                              ? "bg-gradient-to-br from-red-50 to-orange-50 border-red-300"
                              : "bg-gradient-to-br from-green-50 to-emerald-50 border-green-300"
                          }`}
                        >
                          <div className="flex items-center gap-4 mb-4">
                            <div
                              className={`w-14 h-14 rounded-full flex items-center justify-center ${
                                queryResult.is_malicious
                                  ? "bg-gradient-to-r from-red-100 to-orange-100"
                                  : "bg-gradient-to-r from-green-100 to-emerald-100"
                              }`}
                            >
                              <span className="text-2xl">
                                {queryResult.is_malicious ? "⚠️" : "✅"}
                              </span>
                            </div>
                            <div>
                              <h3 className="font-bold text-xl text-gray-900">
                                {queryResult.is_malicious
                                  ? "MALICIOUS"
                                  : "SAFE"}
                              </h3>
                              <p className="text-sm text-gray-600">
                                Confidence:{" "}
                                <span
                                  className={`font-bold ${
                                    queryResult.is_malicious
                                      ? "text-red-600"
                                      : "text-green-600"
                                  }`}
                                >
                                  {queryResult.confidence}%
                                </span>
                              </p>
                            </div>
                          </div>

                          <div className="mb-4 p-3 bg-white rounded-lg border">
                            <p className="text-gray-700">
                              {queryResult.message}
                            </p>
                          </div>

                          {queryResult.detected_patterns &&
                            queryResult.detected_patterns.length > 0 && (
                              <div className="mt-4">
                                <p className="text-sm font-medium text-gray-700 mb-2">
                                  Detected Patterns:
                                </p>
                                <div className="flex flex-wrap gap-2">
                                  {queryResult.detected_patterns.map(
                                    (pattern, idx) => (
                                      <span
                                        key={idx}
                                        className="px-3 py-1.5 bg-red-100 text-red-800 text-xs font-medium rounded-full border border-red-200"
                                      >
                                        {pattern}
                                      </span>
                                    )
                                  )}
                                </div>
                              </div>
                            )}

                          <div className="mt-6 pt-4 border-t border-gray-200">
                            <div className="flex justify-between items-center">
                              <div>
                                <p className="text-xs text-gray-500">
                                  Detection Method
                                </p>
                                <p className="font-semibold text-gray-900">
                                  {queryResult.detection_method || "ml_hybrid"}
                                </p>
                              </div>
                              {queryResult.risk_level && (
                                <div className="text-right">
                                  <p className="text-xs text-gray-500">
                                    Risk Level
                                  </p>
                                  <span
                                    className={`px-3 py-1 rounded-full text-xs font-bold ${
                                      queryResult.risk_level === "critical"
                                        ? "bg-red-100 text-red-800"
                                        : queryResult.risk_level === "high"
                                        ? "bg-orange-100 text-orange-800"
                                        : queryResult.risk_level === "medium"
                                        ? "bg-yellow-100 text-yellow-800"
                                        : "bg-green-100 text-green-800"
                                    }`}
                                  >
                                    {queryResult.risk_level.toUpperCase()}
                                  </span>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="p-8 text-center border-2 border-dashed border-gray-300 rounded-xl bg-gray-50">
                          <div className="w-16 h-16 mx-auto mb-4 bg-blue-100 rounded-full flex items-center justify-center">
                            <span className="text-2xl">🔍</span>
                          </div>
                          <p className="text-gray-600 font-medium">
                            Enter a SQL query to see ML analysis
                          </p>
                          <p className="text-sm text-gray-500 mt-2">
                            Our system will analyze it using 22 features and
                            Random Forest algorithm
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Example Queries */}
              <div className="bg-white p-5 rounded-xl border shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Quick Test Examples
                  </h3>
                  <span className="text-sm text-gray-500">
                    Click any example to test
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {exampleQueries.map((example, idx) => (
                    <button
                      key={idx}
                      onClick={() => tryExampleQuery(example.query)}
                      className={`p-4 text-left rounded-lg border hover:shadow-md transition-all duration-200 ${
                        example.type === "malicious"
                          ? "border-red-200 bg-red-50 hover:bg-red-100 hover:border-red-300"
                          : "border-green-200 bg-green-50 hover:bg-green-100 hover:border-green-300"
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div
                          className={`w-10 h-10 rounded-full flex items-center justify-center ${
                            example.type === "malicious"
                              ? "bg-red-100 text-red-600"
                              : "bg-green-100 text-green-600"
                          }`}
                        >
                          {example.type === "malicious" ? "⚠️" : "✅"}
                        </div>
                        <div className="flex-1">
                          <div className="font-mono text-sm mb-2 line-clamp-2 bg-white/50 p-2 rounded">
                            {example.query}
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-xs text-gray-600">
                              {example.description}
                            </span>
                            <span
                              className={`text-xs px-2 py-1 rounded font-medium ${
                                example.type === "malicious"
                                  ? "bg-red-200 text-red-800"
                                  : "bg-green-200 text-green-800"
                              }`}
                            >
                              {example.type.toUpperCase()}
                            </span>
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}

          {!showQueryChecker && (
            <div className="text-center py-6">
              <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-r from-blue-100 to-indigo-100 rounded-full flex items-center justify-center">
                <span className="text-3xl">🚀</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">
                Live Demo Ready
              </h3>
              <p className="text-gray-600 max-w-2xl mx-auto">
                Click "Try It Now" to test our ML-powered SQL injection
                detection system in real-time. Perfect for demonstrating 99.42%
                accuracy during your thesis defense!
              </p>
            </div>
          )}
        </div>

        {/* Real-time Stats */}
        {stats && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white p-5 rounded-xl shadow-sm border hover:shadow-md transition-shadow">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                  <span className="text-blue-600 text-xl">📊</span>
                </div>
                <div>
                  <div className="text-3xl font-bold text-blue-600">
                    {stats.total_queries}
                  </div>
                  <div className="text-sm text-gray-600">Queries Analyzed</div>
                </div>
              </div>
            </div>
            <div className="bg-white p-5 rounded-xl shadow-sm border hover:shadow-md transition-shadow">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center">
                  <span className="text-red-600 text-xl">⚠️</span>
                </div>
                <div>
                  <div className="text-3xl font-bold text-red-600">
                    {stats.malicious_count}
                  </div>
                  <div className="text-sm text-gray-600">Threats Detected</div>
                </div>
              </div>
            </div>
            <div className="bg-white p-5 rounded-xl shadow-sm border hover:shadow-md transition-shadow">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                  <span className="text-green-600 text-xl">🎯</span>
                </div>
                <div>
                  <div className="text-3xl font-bold text-green-600">
                    {stats.average_confidence}%
                  </div>
                  <div className="text-sm text-gray-600">
                    Detection Accuracy
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {modelInfo && (
          <div className="mt-4 inline-flex items-center px-4 py-3 bg-gradient-to-r from-green-100 to-emerald-100 text-green-800 rounded-full text-sm font-medium border border-green-200 shadow-sm">
            <span className="mr-2">🤖</span>
            <span className="font-semibold">{modelInfo.status}</span>
            <span className="mx-2">•</span>
            <span>Dataset: {modelInfo.dataset_size}</span>
            <span className="mx-2">•</span>
            <span className="font-bold">Accuracy: 99.4%</span>
          </div>
        )}
      </div>

      {/* Stats Grid */}
      {/* <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {dashboardStats.map((stat, index) => (
          <div
            key={index}
            className="bg-white p-6 rounded-xl shadow-sm border hover:shadow-md transition-shadow duration-300"
          >
            <div className="flex items-center justify-between mb-4">
              <div
                className={`w-12 h-12 bg-gradient-to-r ${stat.color} rounded-xl flex items-center justify-center shadow-sm`}
              >
                <span className="text-white text-xl">{stat.icon}</span>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                <p className="text-sm text-gray-500">{stat.label}</p>
              </div>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`bg-gradient-to-r ${stat.color} h-2 rounded-full transition-all duration-500`}
                style={{ width: `${stat.progress}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div> */}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Quick Actions */}
        <div className="bg-white p-6 rounded-xl shadow-sm border">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
            <span className="bg-blue-100 p-2 rounded-lg">🚀</span>
            Start Learning
          </h2>

          <div className="space-y-4">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-5 border border-blue-100 hover:border-blue-200 transition-colors">
              <h3 className="font-semibold text-gray-900 mb-2 text-lg flex items-center gap-2">
                <span>📚</span>
                Interactive Training
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                Learn about different SQL injection techniques through hands-on
                examples and detailed explanations. Understand how our
                22-feature ML model works.
              </p>
              <Link
                to="/training"
                className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-5 py-2.5 rounded-lg inline-block text-center w-full transition-all shadow-md hover:shadow-lg"
              >
                Start Training
              </Link>
            </div>

            <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-5 border border-green-100 hover:border-green-200 transition-colors">
              <h3 className="font-semibold text-gray-900 mb-2 text-lg flex items-center gap-2">
                <span>🎯</span>
                Practice Challenges
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                Test your skills with real-world scenarios and see if you can
                spot the vulnerabilities. Four difficulty levels from Basic to
                Expert.
              </p>
              <Link
                to="/challenges"
                className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white px-5 py-2.5 rounded-lg inline-block text-center w-full transition-all shadow-md hover:shadow-lg"
              >
                Try Challenges
              </Link>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white p-6 rounded-xl shadow-sm border">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
            <span className="bg-purple-100 p-2 rounded-lg">📈</span>
            Recent Activity
          </h2>

          {userProgress && userProgress.recent_detections.length > 0 ? (
            <div className="space-y-4">
              {userProgress.recent_detections
                .slice(0, 4)
                .map((activity, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors border border-gray-200"
                  >
                    <div className="flex items-center space-x-4">
                      <div
                        className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                          activity.is_malicious
                            ? "bg-gradient-to-r from-red-100 to-orange-100"
                            : "bg-gradient-to-r from-green-100 to-emerald-100"
                        }`}
                      >
                        <span
                          className={
                            activity.is_malicious
                              ? "text-red-600 text-lg"
                              : "text-green-600 text-lg"
                          }
                        >
                          {activity.is_malicious ? "⚠️" : "✅"}
                        </span>
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">
                          {activity.is_malicious
                            ? "Malicious Query Detected"
                            : "Safe Query Analyzed"}
                        </p>
                        <div className="flex items-center gap-4 mt-1">
                          <p className="text-sm text-gray-500">
                            <span className="font-medium">Confidence:</span>{" "}
                            {activity.confidence}%
                          </p>
                          <span className="text-xs px-2 py-1 bg-gray-200 text-gray-700 rounded-full">
                            {new Date(activity.timestamp).toLocaleTimeString(
                              [],
                              { hour: "2-digit", minute: "2-digit" }
                            )}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                <span className="text-gray-400 text-2xl">📝</span>
              </div>
              <p className="text-gray-500 font-medium">
                No recent activity yet
              </p>
              <p className="text-sm text-gray-400 mt-2">
                Start using the query analyzer or training modules to see your
                progress here!
              </p>
            </div>
          )}

          {/* Quick Stats */}
          {stats && (
            <div className="mt-6 p-5 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
              <h3 className="font-semibold text-blue-900 mb-3 text-lg">
                System Overview
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="bg-white/70 p-3 rounded-lg">
                  <span className="text-blue-700 font-medium">
                    Total Queries:
                  </span>
                  <span className="float-right font-bold text-blue-800">
                    {stats.total_queries}
                  </span>
                </div>
                <div className="bg-white/70 p-3 rounded-lg">
                  <span className="text-red-700 font-medium">
                    Threats Blocked:
                  </span>
                  <span className="float-right font-bold text-red-600">
                    {stats.malicious_count}
                  </span>
                </div>
                <div className="bg-white/70 p-3 rounded-lg">
                  <span className="text-green-700 font-medium">
                    Success Rate:
                  </span>
                  <span className="float-right font-bold text-green-600">
                    {stats.average_confidence}%
                  </span>
                </div>
                <div className="bg-white/70 p-3 rounded-lg">
                  <span className="text-purple-700 font-medium">
                    Your Level:
                  </span>
                  <span className="float-right font-bold text-purple-800">
                    {userProgress
                      ? getLevel(userProgress.user_info.total_score)
                      : "Beginner"}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer Note */}
      <div className="mt-8 text-center text-sm text-gray-500 border-t pt-6">
        <p>
          <span className="font-medium">SQL Shield Thesis Project</span> •
          Machine Learning-Driven SQL Injection Detection
          <span className="mx-2">•</span>
          99.42% Accuracy using Random Forest with 22 Features
        </p>
        <p className="mt-2 text-xs">
          Ready for Thesis Defense • National University of Computer & Emerging
          Sciences • 2025
        </p>
      </div>
    </div>
  );
};

export default Dashboard;
