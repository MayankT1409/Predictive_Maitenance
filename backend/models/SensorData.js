import mongoose from "mongoose";

const sensorDataSchema = new mongoose.Schema({
  machineId: { type: String, required: true },
  temperature: Number,
  vibration: Number,
  pressure: Number,
  timestamp: { type: Date, default: Date.now }
});

export default mongoose.model("SensorData", sensorDataSchema);
