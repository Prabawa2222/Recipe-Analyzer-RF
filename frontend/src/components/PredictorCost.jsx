import React, { useState } from "react";
import axios from "axios";

const PredictCost = ({ categories }) => {
  const initialFormState = {
    ingredient_cost: "",
    quantity: "",
    waste_percentage: "",
    processing_cost: "",
    labor_hours: "",
    labor_rate: "",
    overhead_cost: "",
    packaging_cost: "",
    recipe_name: "",
    recipe_category: "",
  };

  const [formData, setFormData] = useState(initialFormState);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleCategoryChange = (e) => {
    setFormData({
      ...formData,
      recipe_category: e.target.value,
    });
  };

  const validateForm = () => {
    // Required fields validation
    const requiredFields = [
      "ingredient_cost",
      "quantity",
      "waste_percentage",
      "processing_cost",
    ];
    for (const field of requiredFields) {
      if (!formData[field] || isNaN(parseFloat(formData[field]))) {
        setError(`Please enter a valid number for ${field.replace("_", " ")}`);
        return false;
      }
    }

    // Waste percentage should be between 0 and 100
    if (
      parseFloat(formData.waste_percentage) < 0 ||
      parseFloat(formData.waste_percentage) > 100
    ) {
      setError("Waste percentage must be between 0 and 100");
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Reset states
    setError(null);
    setPrediction(null);
    setSuccess(false);

    // Validate form
    if (!validateForm()) return;

    // Prepare data for API
    const apiData = {};
    for (const [key, value] of Object.entries(formData)) {
      if (value === "") continue;

      // Convert numeric fields to numbers
      if (key !== "recipe_name" && key !== "recipe_category") {
        apiData[key] = parseFloat(value);
      } else {
        apiData[key] = value;
      }
    }

    try {
      setLoading(true);
      const response = await axios.post("/predict", apiData);
      setPrediction(response.data);
      setSuccess(true);
      // Scroll to results
      setTimeout(() => {
        document
          .getElementById("prediction-results")
          ?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } catch (err) {
      console.error("Error making prediction:", err);
      setError(
        err.response?.data?.detail ||
          "Failed to make prediction. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData(initialFormState);
    setPrediction(null);
    setError(null);
    setSuccess(false);
  };

  return (
    <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">
        Predict Recipe Cost
      </h2>

      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-md text-red-600">
          {error}
        </div>
      )}

      {success && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-md text-green-600">
          Prediction successful!
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-8">
        <div className="bg-gray-50 p-6 rounded-md">
          <h3 className="text-lg font-semibold mb-4 text-gray-700">
            Required Information
          </h3>
          <div className="space-y-4">
            <div className="flex flex-col">
              <label
                htmlFor="ingredient_cost"
                className="mb-1 text-gray-600 font-medium"
              >
                Ingredient Cost ($)
              </label>
              <input
                type="number"
                id="ingredient_cost"
                name="ingredient_cost"
                value={formData.ingredient_cost}
                onChange={handleChange}
                step="0.01"
                min="0"
                required
                placeholder="Total cost of all ingredients"
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="flex flex-col">
              <label
                htmlFor="quantity"
                className="mb-1 text-gray-600 font-medium"
              >
                Quantity (Units)
              </label>
              <input
                type="number"
                id="quantity"
                name="quantity"
                value={formData.quantity}
                onChange={handleChange}
                step="0.1"
                min="0"
                required
                placeholder="Number of units produced"
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="flex flex-col">
              <label
                htmlFor="waste_percentage"
                className="mb-1 text-gray-600 font-medium"
              >
                Waste Percentage (%)
              </label>
              <input
                type="number"
                id="waste_percentage"
                name="waste_percentage"
                value={formData.waste_percentage}
                onChange={handleChange}
                step="0.1"
                min="0"
                max="100"
                required
                placeholder="Percentage of waste during production"
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="flex flex-col">
              <label
                htmlFor="processing_cost"
                className="mb-1 text-gray-600 font-medium"
              >
                Processing Cost ($)
              </label>
              <input
                type="number"
                id="processing_cost"
                name="processing_cost"
                value={formData.processing_cost}
                onChange={handleChange}
                step="0.01"
                min="0"
                required
                placeholder="Cost of processing ingredients"
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>
        </div>

        <div className="bg-gray-50 p-6 rounded-md">
          <h3 className="text-lg font-semibold mb-4 text-gray-700">
            Additional Information (Optional)
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div className="flex flex-col">
              <label
                htmlFor="labor_hours"
                className="mb-1 text-gray-600 font-medium"
              >
                Labor Hours
              </label>
              <input
                type="number"
                id="labor_hours"
                name="labor_hours"
                value={formData.labor_hours}
                onChange={handleChange}
                step="0.1"
                min="0"
                placeholder="Hours of labor required"
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="flex flex-col">
              <label
                htmlFor="labor_rate"
                className="mb-1 text-gray-600 font-medium"
              >
                Labor Rate ($/hr)
              </label>
              <input
                type="number"
                id="labor_rate"
                name="labor_rate"
                value={formData.labor_rate}
                onChange={handleChange}
                step="0.01"
                min="0"
                placeholder="Hourly labor rate"
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div className="flex flex-col">
              <label
                htmlFor="overhead_cost"
                className="mb-1 text-gray-600 font-medium"
              >
                Overhead Cost ($)
              </label>
              <input
                type="number"
                id="overhead_cost"
                name="overhead_cost"
                value={formData.overhead_cost}
                onChange={handleChange}
                step="0.01"
                min="0"
                placeholder="Additional overhead costs"
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="flex flex-col">
              <label
                htmlFor="packaging_cost"
                className="mb-1 text-gray-600 font-medium"
              >
                Packaging Cost ($)
              </label>
              <input
                type="number"
                id="packaging_cost"
                name="packaging_cost"
                value={formData.packaging_cost}
                onChange={handleChange}
                step="0.01"
                min="0"
                placeholder="Cost of packaging per unit"
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex flex-col">
              <label
                htmlFor="recipe_name"
                className="mb-1 text-gray-600 font-medium"
              >
                Recipe Name
              </label>
              <input
                type="text"
                id="recipe_name"
                name="recipe_name"
                value={formData.recipe_name}
                onChange={handleChange}
                placeholder="Name of the recipe"
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="flex flex-col">
              <label
                htmlFor="recipe_category"
                className="mb-1 text-gray-600 font-medium"
              >
                Recipe Category
              </label>
              <select
                id="recipe_category"
                name="recipe_category"
                value={formData.recipe_category}
                onChange={handleCategoryChange}
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">Select category (optional)</option>
                {categories.map((category) => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
                <option value="Other">Other</option>
              </select>
            </div>
          </div>
        </div>

        <div className="flex justify-end space-x-4">
          <button
            type="button"
            onClick={handleReset}
            className="px-4 py-2 bg-gray-100 text-gray-800 border border-gray-300 rounded-md font-medium text-sm hover:bg-gray-200 transition-colors"
          >
            Reset
          </button>
          <button
            type="submit"
            className="px-4 py-2 bg-blue-500 text-white rounded-md font-medium text-sm hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={loading}
          >
            {loading ? "Calculating..." : "Calculate Cost"}
          </button>
        </div>
      </form>

      {prediction && (
        <div
          id="prediction-results"
          className="mt-8 bg-blue-50 p-6 rounded-md border border-blue-100"
        >
          <h3 className="text-lg font-semibold mb-4 text-gray-700">
            Prediction Results
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="bg-white p-4 rounded-md shadow-sm">
              <div className="text-sm text-gray-500 mb-1">Cost Per Unit:</div>
              <div className="text-lg font-semibold text-gray-800">
                ${prediction.predicted_cost_per_unit.toFixed(2)}
              </div>
            </div>
            <div className="bg-white p-4 rounded-md shadow-sm">
              <div className="text-sm text-gray-500 mb-1">Suggested Price:</div>
              <div className="text-lg font-semibold text-gray-800">
                ${prediction.suggested_price.toFixed(2)}
              </div>
            </div>
            <div className="bg-white p-4 rounded-md shadow-sm">
              <div className="text-sm text-gray-500 mb-1">Profit Margin:</div>
              <div className="text-lg font-semibold text-gray-800">
                {prediction.profit_margin}%
              </div>
            </div>
            <div className="bg-white p-4 rounded-md shadow-sm">
              <div className="text-sm text-gray-500 mb-1">Model Type:</div>
              <div className="text-lg font-semibold text-gray-800">
                {prediction.model_type}
              </div>
            </div>
            <div className="bg-white p-4 rounded-md shadow-sm">
              <div className="text-sm text-gray-500 mb-1">Model Version:</div>
              <div className="text-lg font-semibold text-gray-800">
                {prediction.model_version}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictCost;
