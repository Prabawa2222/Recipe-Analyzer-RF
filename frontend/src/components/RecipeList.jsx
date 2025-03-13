import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import axios from "axios";

const RecipeList = ({ categories }) => {
  const [recipes, setRecipes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters
  const [categoryFilter, setCategoryFilter] = useState("");
  const [minCost, setMinCost] = useState("");
  const [maxCost, setMaxCost] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const recipesPerPage = 10;

  const fetchRecipes = async () => {
    setLoading(true);
    setError(null);

    try {
      // Build query parameters
      const params = {};
      if (categoryFilter) params.category = categoryFilter;
      if (minCost) params.min_cost = parseFloat(minCost);
      if (maxCost) params.max_cost = parseFloat(maxCost);

      const response = await axios.get("/recipes", { params });
      setRecipes(response.data);
    } catch (err) {
      console.error("Error fetching recipes:", err);
      setError("Failed to load recipes. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecipes();
  }, [categoryFilter, minCost, maxCost]);

  const handleCategoryChange = (e) => {
    setCategoryFilter(e.target.value);
    setCurrentPage(1);
  };

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
    setCurrentPage(1);
  };

  const handleMinCostChange = (e) => {
    setMinCost(e.target.value);
    setCurrentPage(1);
  };

  const handleMaxCostChange = (e) => {
    setMaxCost(e.target.value);
    setCurrentPage(1);
  };

  const clearFilters = () => {
    setCategoryFilter("");
    setMinCost("");
    setMaxCost("");
    setSearchTerm("");
    setCurrentPage(1);
  };

  const handleDeleteRecipe = async (id) => {
    if (window.confirm("Are you sure you want to delete this recipe?")) {
      try {
        await axios.delete(`/recipe/${id}`);
        // Refresh the list
        fetchRecipes();
      } catch (err) {
        console.error("Error deleting recipe:", err);
        setError("Failed to delete recipe. Please try again.");
      }
    }
  };

  // Filter recipes by search term
  const filteredRecipes = recipes.filter(
    (recipe) =>
      recipe.recipe_name &&
      recipe.recipe_name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Pagination
  const indexOfLastRecipe = currentPage * recipesPerPage;
  const indexOfFirstRecipe = indexOfLastRecipe - recipesPerPage;
  const currentRecipes = filteredRecipes.slice(
    indexOfFirstRecipe,
    indexOfLastRecipe
  );
  const totalPages = Math.ceil(filteredRecipes.length / recipesPerPage);

  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
    window.scrollTo(0, 0);
  };

  return (
    <div className="p-5 max-w-full text-black">
      <h2 className="text-xl fornt-bold mb-4">Recipe List</h2>
      <div className="mb-6">
        <div className="flex flex-col md:flex-row md:justify-between md:items-center mb-4 gap-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex flex-col">
              <label
                htmlFor="category-filter"
                className="mb-1 text-gray-600 font-medium"
              >
                Category:
              </label>
              <select
                id="category-filter"
                value={categoryFilter}
                onChange={handleCategoryChange}
                className="w-full md:w-48 px-2 py-2 border bg-white border-gray-300 rounded focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              >
                <option value="">All Categories</option>
                {categories.map((category) => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col">
              <label
                htmlFor="search-filter"
                className="mb-1 text-gray-600 font-medium"
              >
                Search:
              </label>
              <input
                id="search-filter"
                type="text"
                placeholder="Search by name"
                value={searchTerm}
                onChange={handleSearchChange}
                className="w-full md:w-64 px-2 py-2 border bg-white border-gray-300 rounded focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              />
            </div>
          </div>
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex flex-col">
              <label
                htmlFor="min-cost"
                className="mb-1  text-gray-600 font-medium"
              >
                Min Cost ($)
              </label>
              <input
                id="min-cost"
                type="number"
                placeholder="Min"
                value={minCost}
                onChange={handleMinCostChange}
                min="0"
                step="0.01"
                className="w-full bg-white md:w-24 px-2 py-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              />
            </div>
            <div className="flex flex-col">
              <label
                htmlFor="max-cost"
                className="mb-1 text-gray-600 font-medium"
              >
                Max Cost ($):
              </label>
              <input
                id="max-cost"
                type="number"
                placeholder="Max"
                value={maxCost}
                onChange={handleMaxCostChange}
                min="0"
                step="0.01"
                className="w-full bg-white md:w-24 px-2 py-2 border border-gray-300 rounded focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              />
            </div>

            <button
              onClick={clearFilters}
              className="mt-auto py-2 px-4 bg-gray-100 text-gray-800 border border-gray-300 rounded font-medium text-sm hover:bg-gray-200 transition-colors"
            >
              Clear Filters
            </button>
          </div>
        </div>
      </div>
      {error && (
        <div className="p-4 mb-4 bg-red-50 border border-red-200 text-red-600 rounded">
          {error}
        </div>
      )}
      {loading ? (
        <div className="text-center text-gray-600 py-8">Loading recipes...</div>
      ) : (
        <>
          {currentRecipes.length === 0 ? (
            <div className="mb-2 text-sm text-gray-600">
              No recipes found matching your criteria
            </div>
          ) : (
            <>
              <div className="mb-2 text-sm text-gray-600">
                Showing {currentRecipes.length} of {filteredRecipes.length}
                recipes
              </div>
              <div className="overflow-x-auto">
                <table className="w-full bg-white rounded shadow">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left">Recipe Name</th>
                      <th className="px-4 py-3 text-left">Category</th>
                      <th className="px-4 py-3 text-left">Cost Per Unit</th>
                      <th className="px-4 py-3 text-left">Suggested Price</th>
                      <th className="px-4 py-3 text-left">Profit Margin</th>
                      <th className="px-4 py-3 text-left">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {currentRecipes.map((recipe) => (
                      <tr key={recipe.id} className="hover:bg-gray-50">
                        <td className="px-4 py-3">
                          {recipe.recipe_name || "Unamed Recipe"}
                        </td>
                        <td className="px-4 py-3">
                          {recipe.recipe_category || "Uncategorized"}
                        </td>
                        <td className="px-4 py-3">
                          ${recipe.cost_per_unit.toFixed(2)}
                        </td>
                        <td className="px-4 py-3">
                          $
                          {recipe.suggested_price
                            ? recipe.suggested_price.toFixed(2)
                            : "-"}
                        </td>
                        <td className="px-4 py-3">
                          {recipe.profit_margin
                            ? `${recipe.profit_margin}%`
                            : "-"}
                        </td>
                        <td className="px-4 py-3 flex space-x-2">
                          <Link
                            to={`/recipes/${recipe}`}
                            className="inline-block py-1 px-3 bg-white border border-blue-500 text-blue-500 rounded text-sm font-medium hover:bg-blue-50 transition-colors"
                          >
                            View
                          </Link>
                          <button
                            onClick={() => handleDeleteRecipe(recipe.id)}
                            className="py-1 px-3 bg-red-500 text-white rounded text-sm font-medium hover:bg-red-600 transition-colors"
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {totalPages > 1 && (
                <div className="flex justify-between items-center mt-6">
                  <button
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage === 1}
                    className="py-2 px-4 bg-white border border-gray-300 rounded font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    &laquo; Previous
                  </button>

                  <div className="flex space-x-1">
                    {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                      (number) => (
                        <button
                          key={number}
                          onClick={() => handlePageChange(number)}
                          className={`py-2 px-4 rounded text-sm font-medium ${
                            currentPage === number
                              ? "bg-blue-500 text-white"
                              : "bg-white border border-gray-300 hover:bg-gray-100"
                          }`}
                        >
                          {number}
                        </button>
                      )
                    )}
                  </div>

                  <button
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage === totalPages}
                    className="py-2 px-4 bg-white border border-gray-300 rounded font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Next &raquo;
                  </button>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
};

export default RecipeList;
