import express from "express";
import MaintenanceLog from "../models/MaintenanceLog.js";

const router = express.Router();

// POST a new log
router.post("/", async (req, res) => {
  try {
    const log = new MaintenanceLog(req.body);
    await log.save();
    res.status(201).json(log);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// GET all logs
router.get("/", async (req, res) => {
  try {
    const logs = await MaintenanceLog.find().sort({ date: -1 });
    res.json(logs);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

export default router;
