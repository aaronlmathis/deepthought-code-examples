package handlers

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"todo-api/internal/mocks"
	"todo-api/internal/models"
	"todo-api/internal/repository"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTodoHandler_GetTodos(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Create mock database
	mockDB := &mocks.MockTodoDB{}

	// Set up expected behavior
	expectedTodos := []*models.Todo{
		{
			ID:          1,
			Title:       "Test Todo",
			Description: "Test Description",
			Completed:   false,
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		},
	}

	// Configure mock to return expected data
	mockDB.SelectFunc = func(dest interface{}, query string, args ...interface{}) error {
		// Type assert to the expected slice type
		if todos, ok := dest.(*[]*models.Todo); ok {
			*todos = expectedTodos
		}
		return nil
	}

	// Create repository with mock database
	repo := repository.NewTodoRepository(mockDB)
	handler := NewTodoHandler(repo)

	// Create test request
	router := gin.New()
	router.GET("/todos", handler.GetTodos)

	req, _ := http.NewRequest("GET", "/todos", nil)
	w := httptest.NewRecorder()

	// Execute request
	router.ServeHTTP(w, req)

	// Assert results
	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)

	todos := response["todos"].([]interface{})
	assert.Len(t, todos, 1)
}

func TestTodoHandler_CreateTodo(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Create mock database
	mockDB := &mocks.MockTodoDB{}

	// Set up expected behavior for create operation
	expectedTodo := &models.Todo{
		ID:          1,
		Title:       "New Todo",
		Description: "New Description",
		Completed:   false,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	// Configure mock to return created todo
	mockDB.GetFunc = func(dest interface{}, query string, args ...interface{}) error {
		if todo, ok := dest.(*models.Todo); ok {
			*todo = *expectedTodo
		}
		return nil
	}

	// Create repository with mock database
	repo := repository.NewTodoRepository(mockDB)
	handler := NewTodoHandler(repo)

	// Create test request
	router := gin.New()
	router.POST("/todos", handler.CreateTodo)

	todoRequest := models.TodoRequest{
		Title:       "New Todo",
		Description: "New Description",
		Completed:   false,
	}

	jsonData, _ := json.Marshal(todoRequest)
	req, _ := http.NewRequest("POST", "/todos", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	// Execute request
	router.ServeHTTP(w, req)

	// Assert results
	assert.Equal(t, http.StatusCreated, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)

	todo := response["todo"].(map[string]interface{})
	assert.Equal(t, "New Todo", todo["title"])
	assert.Equal(t, "New Description", todo["description"])
	assert.Equal(t, false, todo["completed"])
}

func TestTodoHandler_GetTodo_NotFound(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Create mock database
	mockDB := &mocks.MockTodoDB{}

	// Configure mock to return "not found" error
	mockDB.GetFunc = func(dest interface{}, query string, args ...interface{}) error {
		return sql.ErrNoRows
	}

	// Create repository with mock database
	repo := repository.NewTodoRepository(mockDB)
	handler := NewTodoHandler(repo)

	// Create test request
	router := gin.New()
	router.GET("/todos/:id", handler.GetTodo)

	req, _ := http.NewRequest("GET", "/todos/999", nil)
	w := httptest.NewRecorder()

	// Execute request
	router.ServeHTTP(w, req)

	// Assert results
	assert.Equal(t, http.StatusNotFound, w.Code)
}

func TestTodoHandler_DeleteTodo(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Create mock database
	mockDB := &mocks.MockTodoDB{}

	// Configure mock to simulate successful deletion
	mockDB.ExecFunc = func(query string, args ...interface{}) (sql.Result, error) {
		return mocks.MockResult{RowsAffectedValue: 1}, nil
	}

	// Create repository with mock database
	repo := repository.NewTodoRepository(mockDB)
	handler := NewTodoHandler(repo)

	// Create test request
	router := gin.New()
	router.DELETE("/todos/:id", handler.DeleteTodo)

	req, _ := http.NewRequest("DELETE", "/todos/1", nil)
	w := httptest.NewRecorder()

	// Execute request
	router.ServeHTTP(w, req)

	// Assert results
	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)

	assert.Equal(t, "Todo deleted successfully", response["message"])
}
