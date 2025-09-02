import express from "express";
import axios from "axios"; // to call Python ML API
import Prediction from "../models/Prediction.js";

const router = express.Router();

// POST a new prediction (call Python ML service + save in DB)
router.post("/", async (req, res) => {
  try {
    // Send input data to Python ML API
    const response = await axios.post("http://localhost:5000/predict", req.body);

    // Extract prediction result
    const result = response.data.prediction;

    // Save input + result into MongoDB
    const prediction = new Prediction({
      ...req.body,
      result,
    });

    await prediction.save();

    res.status(201).json(prediction);
  } catch (error) {
    console.error("❌ Prediction error:", error.message);
    res.status(500).json({ error: "Prediction failed" });
  }
});

// GET all predictions
router.get("/", async (req, res) => {
  try {
    const predictions = await Prediction.find().sort({ createdAt: -1 });
    res.json(predictions);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

export default router;
