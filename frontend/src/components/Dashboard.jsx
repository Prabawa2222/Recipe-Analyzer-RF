import axios from "axios";
import React, { useEffect, useState } from "react";
import { Routes, Route, Link } from "react-router-dom";

const Dashboard = ({ statistics }) => {
  const [recentRecipes, setRecentRecipes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isSeedingData, setIsSeedingData] = useState(false);
  const [seedSuccess, setSeedSuccess] = useState(false);

  useEffect(() => {
    const fetchRecentRecipes = async () => {
      try {
        const response = await axios.get("/recipes", { params: { limit: 5 } });
        setRecentRecipes(response.data);
        setError(null);
      } catch (err) {
        console.error("Error fetching recent recipes:", err);
        setError("Failed to load reccent recipes. Please try again later");
      } finally {
        setLoading(false);
      }
    };
    fetchRecentRecipes();
  }, []);

  const handleSeedData = async () => {
    if (
      !window.confirm(
        "This will add sample recipe data to your database. Continue"
      )
    ) {
      return;
    }
    try {
      setIsSeedingData(true);
      setSeedSuccess(false);

      await axios.post("/seed");

      const [statsResponse, recipeResponse] = await Promise.all([
        axios.get("/statistics"),
        axios.get("/recipes", { params: { limit: 5 } }),
      ]);

      setRecentRecipes(recipeResponse.data);
      setSeedSuccess(true);
      setTimeout(() => setSeedSuccess(false), 5000);
    } catch (err) {
      console.error("Error seeding data", err);
      setError("Failed to seed sample data");
    } finally {
      setIsSeedingData(false);
    }
  };

  return (
    <div className="p-5">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-semibold text-black">Dashboard</h2>
        {statistics && statistics.total_recipes === 0 && (
          <button onClick={handleSeedData} disabled={isSeedingData}>
            {isSeedingData ? "Adding Sample Data..." : "Add Sample Data"}
          </button>
        )}
      </div>
      {error && (
        <div className="p-4 bg-red-100 text-red-600 border border-red-300 rounded mb-4">
          {error}
        </div>
      )}
      {seedSuccess && (
        <div className="p-4 bg-green-100 text-green-600 border border-green-300 rounded mb-4">
          Sample data added successfully!
        </div>
      )}
      <div className="space-y-8">
        <div className="bg-white p-6 rounded shadow">
          <h3 className="text-xl font-semibold mb-4 text-black">
            Recipe Statistics
          </h3>
          {statistics ? (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white p-4 rounded shadow border border-gray-200">
                <div className="text-2xl font-bold text-blue-500">
                  {statistics.total_recipes}
                </div>
                <div className="text-gray-600">Total Recipes</div>
              </div>

              <div className="bg-white p-4 rounded shadow border border-gray-200">
                <div className="text-2xl font-bold text-blue-500">
                  $
                  {statistics.average_cost
                    ? statistics.average_cost.toFixed(2)
                    : "0.00"}
                </div>
                <div className="text-gray-600">Average Cost</div>
              </div>

              <div className="bg-white p-4 rounded shadow border border-gray-200">
                <div className="text-2xl font-bold text-blue-500">
                  $
                  {statistics.min_cost
                    ? statistics.min_cost.toFixed(2)
                    : "0.00"}
                </div>
                <div className="text-gray-600">Minimum Cost</div>
              </div>

              <div className="bg-white p-4 rounded shadow border border-gray-200">
                <div className="text-2xl font-bold text-blue-500">
                  $
                  {statistics.max_cost
                    ? statistics.max_cost.toFixed(2)
                    : "0.00"}
                </div>
                <div className="text-gray-600">Maximum Cost</div>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-600 py-4">
              Loading statistics
            </div>
          )}
        </div>
        {statistics &&
          statistics.categories &&
          Object.keys(statistics.categories).length > 0 && (
            <div className="bg-white p-6 rounded shadow">
              <h3 className="text-xl font-semibold mb-4 text-black ">
                Categories
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(statistics.categories).map(
                  ([category, count]) => (
                    <div
                      key={category}
                      className="bg-white p-4 rounded shadow border border-gray-200"
                    >
                      <div className="text-lg font-medium text-black">
                        {category}
                      </div>
                      <div className="text-gray-600">{count} recipes</div>
                    </div>
                  )
                )}
              </div>
            </div>
          )}
        <div className="bg-white p-6 rounded shadow">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl text-black">Recent Recipes</h3>
            <Link to="/recipes" className="text-blue-500 hover:underline">
              View All
            </Link>
          </div>
          {loading ? (
            <div className="texxt-center text-gray-600 py-8">
              Loading recent recipes
            </div>
          ) : recentRecipes.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-gray-50 text-black">
                    <th className="text-left  p-3 border-b">Recipe Name</th>
                    <th className="text-left p-3 border-b">Category</th>
                    <th className="text-left p-3 border-b">Cost Per Unit</th>
                    <th className="text-left p-3 border-b">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {recentRecipes.map((recipe) => (
                    <tr key={recipe.id} className="hover:bg-gray-50 text-black">
                      <td className="p-3 border-b">
                        {recipe.recipe_name || "Unnamed Recipe"}
                      </td>
                      <td className="p-3 border-b">
                        {recipe.recipe_category || "Uncategorized"}
                      </td>
                      <td className="p-3 border-b">
                        ${recipe.cost_per_unit.toFixed(2)}
                      </td>
                      <td className="p-3 border-b">
                        <Link
                          to={`/recipes/${recipe.id}`}
                          className="px-3 py-1 bg-gray-100 text-blue-500 border border-blue-500 rounded hover:bg-blue-50 inline-block text-center"
                        >
                          View
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center bg-white p-8 rounded shadow text-black">
              <p className="mb-4">
                No recipes found. Create your first recipe to get started
              </p>
              <Link
                to="/predict"
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 inline-block"
              >
                Create Recipe
              </Link>
            </div>
          )}
        </div>
        <div className="bg-white p-6 rounded shadow text-black">
          <h3 className="text-xl font-semibold mb-4">Quick Actions</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Link
              to="/predict"
              className="block p-4 bg-white border border-gray-200 rounded shadow hover:shadow-md transition-shadow"
            >
              <div className="text-2xl mb-2">➕</div>
              <div className="font-medium mb-1">Predict New Recipe Cost</div>
              <div className="text-gray-600 text-sm">
                Calculate the cost for a new recipe with your current model
              </div>
            </Link>

            <Link
              to="/recipes"
              className="block p-4 bg-white border border-gray-200 rounded shadow hover:shadow-md transition-shadow"
            >
              <div className="text-2xl mb-2">📋</div>
              <div className="font-medium mb-1">Browse Recipes</div>
              <div className="text-gray-600 text-sm">
                View, edit, and manage all your recipes
              </div>
            </Link>

            <Link
              to="/model"
              className="block p-4 bg-white border border-gray-200 rounded shadow hover:shadow-md transition-shadow"
            >
              <div className="text-2xl mb-2">🔄</div>
              <div className="font-medium mb-1">Retrain Model</div>
              <div className="text-gray-600 text-sm">
                Improve prediction accuracy by retraining with latest data
              </div>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
