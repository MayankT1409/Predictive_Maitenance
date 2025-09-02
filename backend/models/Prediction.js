import mongoose from "mongoose";

const predictionSchema = new mongoose.Schema({
  timestamp: { type: Date, default: Date.now },
  status: { type: String, enum: ["Healthy", "Maintenance Required"] },
  probability: Number,
  sensorSnapshot: Object,
});

export default mongoose.model("Prediction", predictionSchema);
