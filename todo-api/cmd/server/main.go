package main

import (
	"log"
	"os"

	"todo-api/internal/database"
	"todo-api/internal/handlers"
	"todo-api/internal/repository"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

func main() {
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, using system environment variables")
	}

	// Set Gin mode from environment
	if ginMode := os.Getenv("GIN_MODE"); ginMode != "" {
		gin.SetMode(ginMode)
	}

	// Load database configuration
	dbConfig := database.LoadConfig()

	// Connect to database
	db, err := database.NewConnection(dbConfig)
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}
	defer db.Close()

	// Initialize repository and handler
	todoRepo := repository.NewTodoRepository(db)
	todoHandler := handlers.NewTodoHandler(todoRepo)

	// Setup router
	router := gin.Default()

	// Health check with database connectivity
	router.GET("/health", func(c *gin.Context) {
		if err := db.HealthCheck(); err != nil {
			c.JSON(500, gin.H{"status": "unhealthy", "error": err.Error()})
			return
		}

		stats := db.GetStats()
		c.JSON(200, gin.H{
			"status": "healthy",
			"database": gin.H{
				"connected":        true,
				"open_connections": stats.OpenConnections,
				"in_use":           stats.InUse,
				"idle":             stats.Idle,
			},
		})
	})

	// API routes
	v1 := router.Group("/api/v1")
	{
		todos := v1.Group("/todos")
		{
			todos.GET("", todoHandler.GetTodos)
			todos.POST("", todoHandler.CreateTodo)
			todos.GET("/:id", todoHandler.GetTodo)
			todos.PUT("/:id", todoHandler.UpdateTodo)
			todos.DELETE("/:id", todoHandler.DeleteTodo)
			todos.GET("/stats", todoHandler.GetStats)
		}
	}

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Starting Todo API server on :%s", port)
	log.Printf("Database: %s@%s:%d/%s", dbConfig.User, dbConfig.Host, dbConfig.Port, dbConfig.DBName)
	log.Println("API endpoints available at http://localhost:" + port + "/api/v1/todos")

	if err := router.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}
