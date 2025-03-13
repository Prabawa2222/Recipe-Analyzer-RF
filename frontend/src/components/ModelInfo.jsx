import React, { useState, useEffect } from "react";
import axios from "axios";

const ModelInfo = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retraining, setRetraining] = useState(false);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      setLoading(true);
      const response = await axios.get("/model/info");
      setModelInfo(response.data);
      setError(null);
    } catch (err) {
      console.error("Error fetching model information:", err);
      setError("Failed to load model information. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  const handleRetrainModel = async () => {
    if (
      !window.confirm(
        "Are you sure you want to retrain the model? This may take some time."
      )
    ) {
      return;
    }

    try {
      setRetraining(true);
      setSuccess(false);
      setError(null);

      const response = await axios.post("/model/retrain");
      setModelInfo(response.data.model_info);
      setSuccess(true);

      // Scroll to notification
      setTimeout(() => {
        document
          .getElementById("notification")
          ?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } catch (err) {
      console.error("Error retraining model:", err);
      setError("Failed to retrain model. Please try again later.");
    } finally {
      setRetraining(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return "Unknown";

    try {
      const date = new Date(dateString);
      return date.toLocaleDateString() + " " + date.toLocaleTimeString();
    } catch (e) {
      return dateString;
    }
  };

  return (
    <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-md">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">Model Information</h2>
        <button
          onClick={handleRetrainModel}
          disabled={retraining}
          className="px-4 py-2 bg-blue-500 text-white rounded-md font-medium text-sm hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {retraining ? "Retraining..." : "Retrain Model"}
        </button>
      </div>

      {error && (
        <div
          id="notification"
          className="p-4 mb-6 bg-red-50 border border-red-200 rounded-md text-red-600"
        >
          {error}
        </div>
      )}

      {success && (
        <div
          id="notification"
          className="p-4 mb-6 bg-green-50 border border-green-200 rounded-md text-green-600"
        >
          Model retrained successfully!
        </div>
      )}

      {loading ? (
        <div className="text-center text-gray-600 py-12">
          Loading model information...
        </div>
      ) : modelInfo ? (
        <div className="space-y-6">
          <div className="bg-gray-50 p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold mb-4 text-gray-700">
              Current Model
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex flex-col">
                <span className="text-sm text-gray-500 mb-1">Model Type:</span>
                <span className="font-medium text-gray-800">
                  {modelInfo.current_model?.charAt(0).toUpperCase() +
                    modelInfo.current_model?.slice(1).replace(/_/g, " ")}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-sm text-gray-500 mb-1">Version:</span>
                <span className="font-medium text-gray-800">
                  {modelInfo.version || "Unknown"}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-sm text-gray-500 mb-1">
                  Last Trained:
                </span>
                <span className="font-medium text-gray-800">
                  {formatDate(modelInfo.last_trained)}
                </span>
              </div>
            </div>
          </div>

          {modelInfo.metrics && (
            <div className="bg-gray-50 p-6 rounded-lg shadow-sm">
              <h3 className="text-lg font-semibold mb-4 text-gray-700">
                Model Performance
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex flex-col">
                  <span className="text-sm text-gray-500 mb-1">
                    Mean Absolute Error:
                  </span>
                  <span className="font-medium text-gray-800">
                    {modelInfo.metrics.mae
                      ? modelInfo.metrics.mae.toFixed(4)
                      : "N/A"}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-sm text-gray-500 mb-1">
                    Mean Squared Error:
                  </span>
                  <span className="font-medium text-gray-800">
                    {modelInfo.metrics.mse
                      ? modelInfo.metrics.mse.toFixed(4)
                      : "N/A"}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-sm text-gray-500 mb-1">R² Score:</span>
                  <span className="font-medium text-gray-800">
                    {modelInfo.metrics.r2
                      ? modelInfo.metrics.r2.toFixed(4)
                      : "N/A"}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-sm text-gray-500 mb-1">
                    Cross Validation Score:
                  </span>
                  <span className="font-medium text-gray-800">
                    {modelInfo.metrics.cv_score
                      ? modelInfo.metrics.cv_score.toFixed(4)
                      : "N/A"}
                  </span>
                </div>
              </div>
            </div>
          )}

          <div className="bg-gray-50 p-6 rounded-lg shadow-sm">
            <h3 className="text-lg font-semibold mb-4 text-gray-700">
              Features Used
            </h3>
            <div>
              {modelInfo.features && modelInfo.features.length > 0 ? (
                <ul className="list-disc pl-5 space-y-1">
                  {modelInfo.features.map((feature, index) => (
                    <li key={index} className="text-gray-700">
                      {feature.replace(/_/g, " ")}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-gray-600">
                  No feature information available
                </p>
              )}
            </div>
          </div>

          <div className="bg-blue-50 p-6 rounded-lg border border-blue-100">
            <h3 className="text-lg font-semibold mb-3 text-gray-700">
              About the Model
            </h3>
            <div className="space-y-3 text-gray-700">
              <p>
                This prediction model uses machine learning to estimate recipe
                costs based on input parameters. The system automatically
                selects the best algorithm from multiple models (Random Forest,
                Gradient Boosting, and Linear Regression) based on
                cross-validation performance.
              </p>
              <p>
                When you retrain the model, the system will use all available
                recipe data to improve prediction accuracy. This is recommended
                when you have added several new recipes or notice that
                predictions are becoming less accurate.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center bg-gray-50 p-8 rounded-lg shadow-sm text-gray-600">
          No model information available
        </div>
      )}
    </div>
  );
};

export default ModelInfo;
