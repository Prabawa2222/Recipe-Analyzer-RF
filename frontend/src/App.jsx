import { useState, useEffect } from "react";
import { Routes, Route, Link } from "react-router-dom";
import axios from "axios";
import "./App.css";
import Dashboard from "./components/Dashboard";
import RecipeDetail from "./components/RecipeDetail";
import RecipeList from "./components/RecipeList";
import PredictorCost from "./components/PredictorCost";
import ModelInfo from "./components/ModelInfo";

const API_URL = "http://localhost:8000";
axios.defaults.baseURL = API_URL;

function App() {
  const [statistics, setStatistics] = useState(null);
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [statsResponse, categoriesResponse] = await Promise.all([
          axios.get("/statistics"),
          axios.get("/categories"),
        ]);
        setStatistics(statsResponse.data);
        setCategories(categoriesResponse.data);
      } catch (error) {
        console.error("Error fetching initial data:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchInitialData();
  }, []);

  return (
    <div className="min-h-screen ">
      <header className="bg-black shadow-md p-5">
        <div className="container mx-auto flex justify-between items-center ">
          <h1 className="text-2xl">Recipe Cost Predictor</h1>
          <nav className="flex space-x-6">
            <Link to="/">Dashboard</Link>
            <Link to="/predict">Predict Cost</Link>
            <Link to="/recipes">Recipes</Link>
            <Link to="/model">Model Info</Link>
          </nav>
        </div>
      </header>
      <main className="min-h-screen">
        {loading ? (
          <div>Loading</div>
        ) : (
          <Routes>
            <Route path="/" element={<Dashboard statistics={statistics} />} />
            <Route
              path="/predict"
              element={<PredictorCost categories={categories} />}
            />
            <Route
              path="/recipes"
              element={<RecipeList categories={categories} />}
            />
            <Route path="/recipes/:id" element={<RecipeDetail />} />
            <Route path="/model" element={<ModelInfo />} />
          </Routes>
        )}
      </main>
      <footer>
        <p>&copy; {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;
