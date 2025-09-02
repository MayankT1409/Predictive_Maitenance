import express from "express";
import cors from "cors";
import sensorRoutes from "./routes/sensorRoutes.js";
import predictionRoutes from "./routes/predictionRoutes.js";
import maintenanceRoutes from "./routes/maintenanceRoutes.js";

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use("/api/sensors", sensorRoutes);
app.use("/api/predictions", predictionRoutes);
app.use("/api/maintenance", maintenanceRoutes);

// Health check route
app.get("/", (req, res) => {
  res.send("Predictive Maintenance API is running...");
});

export default app;
