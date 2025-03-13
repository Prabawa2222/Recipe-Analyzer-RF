import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";

const RecipeDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();

  const [recipe, setRecipe] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [editMode, setEditMode] = useState(false);
  const [formData, setFormData] = useState({});
  const [profitMargin, setProfitMargin] = useState("");
  const [savingMargin, setSavingMargin] = useState(false);
  const [savingRecipe, setSavingRecipe] = useState(false);

  useEffect(() => {
    const fetchRecipe = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`/recipe/${id}`);
        setRecipe(response.data);
        setFormData(response.data);
        setProfitMargin(response.data.profitMargin || "");
      } catch (err) {
        console.error("Error fetching recipe details:", err);
        setError("Failed to load recipe details. Please try again later.");
      } finally {
        setLoading(false);
      }
    };
    fetchRecipe();
  }, [id]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]:
        name === "recipe_name" || name === "recipe_category"
          ? value
          : parseFloat(value),
    });
  };

  const handleMarginChange = (e) => {
    setProfitMargin(e.target.value);
  };

  const handleSaveMargin = async () => {
    if (!profitMargin || isNaN(parseFloat(profitMargin))) {
      alert("Please enter a valid profit margin percentage");
      return;
    }

    try {
      setSavingMargin(true);
      const response = await axios.post("/calculate-price", {
        recipe_id: id,
        target_margin: parseFloat(profitMargin),
      });

      //Update recipe with new price and margin
      setRecipe({
        ...recipe,
        profit_margin: response.data.profit_margin,
        suggested_price: response.data.suggested_price,
      });
    } catch (err) {
      console.error("Error updating profit margin:", err);
      alert("Failed to update profit margin. Please try again.");
    } finally {
      setSavingMargin(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      setSavingRecipe(true);
      const response = await axios.put(`/recipe/${id}`, formData);
      setRecipe(response.data);
      setEditMode(false);
      alert("Recipe updated successfully");
    } catch (err) {
      console.error("Error updating recipe:", err);
      alert("Failed to update recipe. Please try again.");
    } finally {
      setSavingRecipe(false);
    }
  };

  const handleDelete = async () => {
    if (window.confirm("Are you sure you want to delete this recipe?")) {
      try {
        await axios.delete(`/recipe/${id}`);
        alert("Recipe deleted sucessfully");
        navigate("/recipes");
      } catch (err) {
        console.error("Error deleting recipe:", err);
        alert("Failed to delete recipe. Please try again.");
      }
    }
  };

  if (loading) {
    return (
      <div className="text-center text-gray-600 py-8">
        Loading recipe details...
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-100 text-red-600 border border-red-300 rounded mb-4">
        {error}
      </div>
    );
  }

  if (!recipe) {
    return (
      <div className="p-4 bg-gray-100 text-gray-600 border border-gray-300 rounded mb-4">
        Recipe not found.
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-4 text-black">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-semibold">
          {recipe.recipe_name || "Unnamed Recipe"}
        </h2>
        <div className="flex gap-2">
          <button
            onClick={() => setEditMode(!editMode)}
            className={`px-4 py-2 rounded font-medium text-sm transition-colors ${
              editMode
                ? "bg-gray-100 text-gray-800 border border-gray-300 hover:bg-gray-200"
                : "bg-blue-500 text-white hover:bg-blue-600"
            }`}
          >
            {editMode ? "Cancel Edit" : "Edit Recipe"}
          </button>
          <button
            onClick={handleDelete}
            className="px-4 py-2 bg-red-600 text-white rounded font-medium text-sm hover:bg-red-700 transition-colors"
          >
            Delete Recipe
          </button>
        </div>
      </div>

      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded shadow">
            <h3 className="text-xl font-semibold mb-4">Cost Summary</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="border-b pb-2">
                <span className="text-gray-600 block">Cost Per Unit:</span>
                <span className="text-lg font-medium">
                  ${recipe.cost_per_unit.toFixed(2)}
                </span>
              </div>
              <div className="border-b pb-2">
                <span className="text-gray-600 block">Suggested Price:</span>
                <span className="text-lg font-medium">
                  $
                  {recipe.suggested_price
                    ? recipe.suggested_price.toFixed(2)
                    : "-"}
                </span>
              </div>
              <div className="border-b pb-2">
                <span className="text-gray-600 block">Profit Margin:</span>
                <span className="text-lg font-medium">
                  {recipe.profit_margin ? `${recipe.profit_margin}%` : "-"}
                </span>
              </div>
              <div className="border-b pb-2">
                <span className="text-gray-600 block">Category:</span>
                <span className="text-lg font-medium">
                  {recipe.recipe_category || "Uncategorized"}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded shadow">
            <h3 className="text-xl font-semibold mb-4">
              Profit Margin Calculator
            </h3>
            <div className="space-y-4">
              <div className="mb-4">
                <label
                  htmlFor="profit-margin"
                  className="block mb-1 font-medium text-gray-600"
                >
                  Target Profit Margin (%):
                </label>
                <input
                  type="number"
                  id="profit-margin"
                  value={profitMargin}
                  onChange={handleMarginChange}
                  step="0.1"
                  min="0"
                  max="100"
                  placeholder="Enter profit margin %"
                  className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                />
              </div>
              <button
                onClick={handleSaveMargin}
                disabled={savingMargin}
                className="px-4 py-2 bg-blue-500 text-white rounded font-medium text-sm hover:bg-blue-600 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {savingMargin ? "Updating..." : "Update Price"}
              </button>
            </div>
          </div>
        </div>

        {editMode ? (
          <form onSubmit={handleSubmit} className="bg-white p-6 rounded shadow">
            <h3 className="text-xl font-semibold mb-4">Edit Recipe Details</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label
                  htmlFor="recipe_name"
                  className="block mb-1 font-medium text-gray-600"
                >
                  Recipe Name
                </label>
                <input
                  type="text"
                  id="recipe_name"
                  name="recipe_name"
                  value={formData.recipe_name || ""}
                  onChange={handleChange}
                  placeholder="Recipe Name"
                  className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                />
              </div>

              <div>
                <label
                  htmlFor="recipe_category"
                  className="block mb-1 font-medium text-gray-600"
                >
                  Category
                </label>
                <input
                  type="text"
                  id="recipe_category"
                  name="recipe_category"
                  value={formData.recipe_category || ""}
                  onChange={handleChange}
                  placeholder="Category"
                  className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label
                  htmlFor="ingredient_cost"
                  className="block mb-1 font-medium text-gray-600"
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
                  className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                />
              </div>

              <div>
                <label
                  htmlFor="quantity"
                  className="block mb-1 font-medium text-gray-600"
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
                  className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                />
              </div>
            </div>

            <div className="flex justify-end gap-4 mt-6">
              <button
                type="button"
                onClick={() => setEditMode(false)}
                className="px-4 py-2 bg-gray-100 text-gray-800 border border-gray-300 rounded hover:bg-gray-200 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={savingRecipe}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {savingRecipe ? "Saving..." : "Save Changes"}
              </button>
            </div>
          </form>
        ) : (
          // You can add recipe details display section here if needed
          <div className="bg-white p-6 rounded shadow">
            <h3 className="text-xl font-semibold mb-4">Recipe Details</h3>
            {/* Additional recipe details would go here */}
          </div>
        )}
      </div>
    </div>
  );
};

export default RecipeDetail;
