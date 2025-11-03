import axios from "axios";

const API_BASE_URL = "http://localhost:8000/api";

// Create axios instance with better error handling
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Add response interceptor for better error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error("API Error:", error);
    if (error.response) {
      // Server responded with error status
      throw new Error(
        error.response.data.detail || `Server error: ${error.response.status}`
      );
    } else if (error.request) {
      // Network error
      throw new Error("Network error: Unable to connect to server");
    } else {
      // Other error
      throw new Error("Request configuration error");
    }
  }
);

export const checkSQLInjection = async (query) => {
  try {
    const response = await api.post("/detect", {
      query: query,
      context: "training",
    });
    return response;
  } catch (error) {
    console.error("Detection API Error:", error);
    throw error;
  }
};

export const submitChallenge = async (challengeId, query) => {
  try {
    const response = await api.post("/submit-challenge", {
      challenge_id: challengeId,
      query: query,
      user_id: 1,
    });
    return response;
  } catch (error) {
    console.error("Challenge API Error:", error);
    throw error;
  }
};

export const getDetectionStats = async () => {
  try {
    const response = await api.get("/stats");
    return response;
  } catch (error) {
    console.error("Stats API Error:", error);
    throw error;
  }
};

export const getChallengeInfo = async () => {
  try {
    const response = await api.get("/challenge-info");
    return response;
  } catch (error) {
    console.error("Challenge Info API Error:", error);
    throw error;
  }
};

export const getModelInfo = async () => {
  try {
    const response = await api.get("/model-info");
    return response;
  } catch (error) {
    console.error("Model Info API Error:", error);
    throw error;
  }
};

export const getUserStats = async (userId = 1) => {
  try {
    const response = await api.get(`/user-stats/${userId}`);
    return response;
  } catch (error) {
    console.error("User Stats API Error:", error);
    throw error;
  }
};
