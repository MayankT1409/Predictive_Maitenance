import mongoose from "mongoose";

const maintenanceLogSchema = new mongoose.Schema({
  machineId: { type: String, required: true },
  actionTaken: { type: String, required: true },
  technician: String,
  date: { type: Date, default: Date.now }
});

export default mongoose.model("MaintenanceLog", maintenanceLogSchema);
