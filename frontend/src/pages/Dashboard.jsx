import React from "react";
import { Link } from "react-router-dom";

const Dashboard = () => {
  const stats = [
    {
      label: "Training Completed",
      value: "3/10",
      progress: 30,
      color: "from-green-500 to-emerald-500",
      icon: "✅",
    },
    {
      label: "Challenges Solved",
      value: "5",
      progress: 50,
      color: "from-blue-500 to-cyan-500",
      icon: "⚡",
    },
    {
      label: "Current Level",
      value: "Beginner",
      progress: 25,
      color: "from-purple-500 to-pink-500",
      icon: "📊",
    },
    {
      label: "Total Points",
      value: "250",
      progress: 20,
      color: "from-orange-500 to-red-500",
      icon: "⭐",
    },
  ];

  const recentActivity = [
    {
      action: "Completed Basic SQL Injection Training",
      points: 50,
      time: "2 hours ago",
    },
    { action: "Solved Login Bypass Challenge", points: 30, time: "1 day ago" },
    { action: "Earned Quick Learner Badge", points: 20, time: "2 days ago" },
  ];

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
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, index) => (
          <div key={index} className="stat-card">
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
        <div className="card p-6">
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
                className="btn-primary inline-block text-center w-full"
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
                className="btn-primary inline-block text-center w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
              >
                Try Challenges
              </Link>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="card p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Recent Activity
          </h2>
          <div className="space-y-4">
            {recentActivity.map((activity, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                    <span className="text-green-600">⭐</span>
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">
                      {activity.action}
                    </p>
                    <p className="text-sm text-gray-500">{activity.time}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-bold text-green-600">+{activity.points}</p>
                  <p className="text-xs text-gray-500">XP</p>
                </div>
              </div>
            ))}
          </div>

          {/* Badges Section */}
          <div className="mt-6">
            <h3 className="font-semibold text-gray-900 mb-4">Your Badges</h3>
            <div className="flex space-x-4">
              <div className="text-center">
                <div className="w-16 h-16 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-2">
                  <span className="text-2xl">🚀</span>
                </div>
                <p className="text-xs font-medium">Quick Start</p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-2">
                  <span className="text-2xl">📚</span>
                </div>
                <p className="text-xs font-medium">Learner</p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-2">
                  <span className="text-2xl">⚡</span>
                </div>
                <p className="text-xs font-medium">Fast Solver</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
