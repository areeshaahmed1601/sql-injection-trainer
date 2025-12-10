import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { getDetectionStats, getUserProgress, getModelInfo } from "../utils/api";

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [userProgress, setUserProgress] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

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

        {/* Real-time Stats */}
        {stats && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white p-4 rounded-lg shadow-sm border text-center">
              <div className="text-2xl font-bold text-blue-600">
                {stats.total_queries}
              </div>
              <div className="text-sm text-gray-600">Queries Analyzed</div>
            </div>
            <div className="bg-white p-4 rounded-lg shadow-sm border text-center">
              <div className="text-2xl font-bold text-red-600">
                {stats.malicious_count}
              </div>
              <div className="text-sm text-gray-600">Threats Detected</div>
            </div>
            <div className="bg-white p-4 rounded-lg shadow-sm border text-center">
              <div className="text-2xl font-bold text-green-600">
                {stats.average_confidence}%
              </div>
              <div className="text-sm text-gray-600">Detection Accuracy</div>
            </div>
          </div>
        )}

        {modelInfo && (
          <div className="mt-4 inline-flex items-center px-4 py-2 bg-gradient-to-r from-green-100 to-emerald-100 text-green-800 rounded-full text-sm font-medium border border-green-200">
            🎯 {modelInfo.status} | Dataset: {modelInfo.dataset_size} |
            Accuracy: 99.4%
          </div>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {dashboardStats.map((stat, index) => (
          <div
            key={index}
            className="bg-white p-6 rounded-xl shadow-sm border hover:shadow-md transition-shadow"
          >
            <div className="flex items-center justify-between mb-4">
              <div
                className={`w-12 h-12 bg-gradient-to-r ${stat.color} rounded-xl flex items-center justify-center`}
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
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Quick Actions */}
        <div className="bg-white p-6 rounded-xl shadow-sm border">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Start Learning
          </h2>

          <div className="space-y-4">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-4 border border-blue-100">
              <h3 className="font-semibold text-gray-900 mb-2">
                🚀 Interactive Training
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                Learn about different SQL injection techniques through hands-on
                examples and detailed explanations.
              </p>
              <Link
                to="/training"
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg inline-block text-center w-full transition-colors"
              >
                Start Training
              </Link>
            </div>

            <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-4 border border-green-100">
              <h3 className="font-semibold text-gray-900 mb-2">
                🎯 Practice Challenges
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                Test your skills with real-world scenarios and see if you can
                spot the vulnerabilities.
              </p>
              <Link
                to="/challenges"
                className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg inline-block text-center w-full transition-colors"
              >
                Try Challenges
              </Link>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white p-6 rounded-xl shadow-sm border">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Recent Activity
          </h2>

          {userProgress && userProgress.recent_detections.length > 0 ? (
            <div className="space-y-4">
              {userProgress.recent_detections
                .slice(0, 3)
                .map((activity, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors"
                  >
                    <div className="flex items-center space-x-3">
                      <div
                        className={`w-10 h-10 rounded-full flex items-center justify-center ${
                          activity.is_malicious ? "bg-red-100" : "bg-green-100"
                        }`}
                      >
                        <span
                          className={
                            activity.is_malicious
                              ? "text-red-600"
                              : "text-green-600"
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
                        <p className="text-sm text-gray-500">
                          Confidence: {activity.confidence}%
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-gray-500">No recent activity yet</p>
              <p className="text-sm text-gray-400 mt-2">
                Start training to see your progress here!
              </p>
            </div>
          )}

          {/* Quick Stats */}
          {stats && (
            <div className="mt-6 p-4 bg-blue-50 rounded-xl">
              <h3 className="font-semibold text-blue-900 mb-2">
                System Overview
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-blue-700">Total Queries:</span>
                  <span className="float-right font-semibold">
                    {stats.total_queries}
                  </span>
                </div>
                <div>
                  <span className="text-blue-700">Threats Blocked:</span>
                  <span className="float-right font-semibold text-red-600">
                    {stats.malicious_count}
                  </span>
                </div>
                <div>
                  <span className="text-blue-700">Success Rate:</span>
                  <span className="float-right font-semibold text-green-600">
                    {stats.average_confidence}%
                  </span>
                </div>
                <div>
                  <span className="text-blue-700">Your Level:</span>
                  <span className="float-right font-semibold">
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
    </div>
  );
};

export default Dashboard;
