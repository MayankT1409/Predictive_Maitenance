import express from "express";
import SensorData from "../models/SensorData.js";

const router = express.Router();

// POST new sensor data
router.post("/", async (req, res) => {
  try {
    const data = new SensorData(req.body);
    await data.save();
    res.status(201).json(data);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// GET all sensor data
router.get("/", async (req, res) => {
  try {
    const data = await SensorData.find().sort({ timestamp: -1 });
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

export default router;
