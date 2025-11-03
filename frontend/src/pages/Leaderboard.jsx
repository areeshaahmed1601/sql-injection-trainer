import React from "react";

const Leaderboard = () => {
  const leaderboardData = [
    {
      rank: 1,
      name: "Alice Johnson",
      score: 950,
      completed: 10,
      level: "Expert",
      avatar: "👩‍💻",
    },
    {
      rank: 2,
      name: "Bob Smith",
      score: 870,
      completed: 9,
      level: "Advanced",
      avatar: "👨‍💻",
    },
    {
      rank: 3,
      name: "Carol Davis",
      score: 820,
      completed: 8,
      level: "Advanced",
      avatar: "👩‍🎓",
    },
    {
      rank: 4,
      name: "David Wilson",
      score: 780,
      completed: 8,
      level: "Intermediate",
      avatar: "👨‍🎓",
    },
    {
      rank: 5,
      name: "Eva Brown",
      score: 720,
      completed: 7,
      level: "Intermediate",
      avatar: "👩‍🔬",
    },
    {
      rank: 6,
      name: "Frank Miller",
      score: 680,
      completed: 7,
      level: "Intermediate",
      avatar: "👨‍🔬",
    },
    {
      rank: 7,
      name: "Grace Lee",
      score: 620,
      completed: 6,
      level: "Beginner",
      avatar: "👩‍💼",
    },
    {
      rank: 8,
      name: "Henry Clark",
      score: 580,
      completed: 6,
      level: "Beginner",
      avatar: "👨‍💼",
    },
  ];

  const userStats = {
    rank: 15,
    score: 250,
    completed: 3,
    level: "Beginner",
    nextLevel: "Intermediate",
    pointsNeeded: 250,
  };

  const badges = [
    { name: "Quick Start", icon: "🚀", earned: true },
    { name: "SQL Master", icon: "🏆", earned: false },
    { name: "Fast Solver", icon: "⚡", earned: true },
    { name: "Perfect Score", icon: "⭐", earned: false },
    { name: "Marathon Runner", icon: "🏃", earned: false },
    { name: "Early Bird", icon: "🐦", earned: true },
  ];

  return (
    <div className="max-w-6xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Leaderboard</h1>
        <p className="text-xl text-gray-600">
          Compete with other learners and climb your way to the top!
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Leaderboard Table */}
        <div className="lg:col-span-2">
          <div className="card overflow-hidden">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6">
              <h2 className="text-2xl font-bold text-white">Top Performers</h2>
              <p className="text-blue-100">
                See who's leading the SQL injection mastery
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Rank
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Player
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Level
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Score
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Completed
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {leaderboardData.map((player) => (
                    <tr
                      key={player.rank}
                      className="hover:bg-gray-50 transition-colors"
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`inline-flex items-center justify-center w-8 h-8 rounded-full ${
                            player.rank === 1
                              ? "bg-yellow-100 text-yellow-800"
                              : player.rank === 2
                              ? "bg-gray-100 text-gray-800"
                              : player.rank === 3
                              ? "bg-orange-100 text-orange-800"
                              : "bg-blue-100 text-blue-800"
                          } font-semibold`}
                        >
                          {player.rank}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center space-x-3">
                          <span className="text-2xl">{player.avatar}</span>
                          <div>
                            <p className="text-sm font-medium text-gray-900">
                              {player.name}
                            </p>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            player.level === "Expert"
                              ? "bg-purple-100 text-purple-800"
                              : player.level === "Advanced"
                              ? "bg-red-100 text-red-800"
                              : player.level === "Intermediate"
                              ? "bg-blue-100 text-blue-800"
                              : "bg-green-100 text-green-800"
                          }`}
                        >
                          {player.level}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-gray-900">
                        {player.score}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {player.completed} challenges
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* User Stats & Badges */}
        <div className="space-y-6">
          {/* User Stats */}
          <div className="card p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Your Stats</h2>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl">
                <div>
                  <p className="text-sm text-gray-600">Current Rank</p>
                  <p className="text-2xl font-bold text-gray-900">
                    #{userStats.rank}
                  </p>
                </div>
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                  <span className="text-white font-bold">🏆</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-gray-50 rounded-xl">
                  <p className="text-2xl font-bold text-yellow-600">
                    {userStats.score}
                  </p>
                  <p className="text-sm text-gray-600">Total Points</p>
                </div>
                <div className="text-center p-4 bg-gray-50 rounded-xl">
                  <p className="text-2xl font-bold text-green-600">
                    {userStats.completed}
                  </p>
                  <p className="text-sm text-gray-600">Challenges</p>
                </div>
              </div>

              {/* Level Progress */}
              <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl">
                <div className="flex justify-between text-sm mb-2">
                  <span className="font-medium text-gray-700">
                    {userStats.level}
                  </span>
                  <span className="text-gray-600">
                    {userStats.pointsNeeded} to next level
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${
                        (userStats.score /
                          (userStats.score + userStats.pointsNeeded)) *
                        100
                      }%`,
                    }}
                  ></div>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Next: {userStats.nextLevel}
                </p>
              </div>
            </div>
          </div>

          {/* Badges */}
          <div className="card p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">
              Your Badges
            </h2>
            <div className="grid grid-cols-2 gap-4">
              {badges.map((badge, index) => (
                <div
                  key={index}
                  className={`text-center p-4 rounded-xl border-2 transition-all ${
                    badge.earned
                      ? "bg-gradient-to-br from-yellow-50 to-amber-50 border-yellow-200"
                      : "bg-gray-50 border-gray-200 opacity-50"
                  }`}
                >
                  <div
                    className={`text-2xl mb-2 ${
                      badge.earned ? "" : "grayscale"
                    }`}
                  >
                    {badge.icon}
                  </div>
                  <p
                    className={`text-sm font-medium ${
                      badge.earned ? "text-gray-900" : "text-gray-400"
                    }`}
                  >
                    {badge.name}
                  </p>
                  {badge.earned && (
                    <span className="inline-block w-2 h-2 bg-green-500 rounded-full mt-1"></span>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="card p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">
              Keep Learning
            </h2>
            <div className="space-y-3">
              <button className="w-full btn-primary">Continue Training</button>
              <button className="w-full btn-secondary">
                Try More Challenges
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Leaderboard;
