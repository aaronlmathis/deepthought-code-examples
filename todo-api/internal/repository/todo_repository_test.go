package repository

import (
	"os"
	"testing"
	"time"

	"todo-api/internal/database"
	"todo-api/internal/models"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var testDB *database.DB

func TestMain(m *testing.M) {
	// Setup test database
	config := database.Config{
		Host:            "localhost",
		Port:            5433,
		User:            "todouser",
		Password:        "todopass",
		DBName:          "todoapi_test",
		SSLMode:         "disable",
		MaxOpenConns:    5,
		MaxIdleConns:    2,
		ConnMaxLifetime: time.Minute,
		ConnMaxIdleTime: time.Minute,
	}

	var err error
	testDB, err = database.NewConnection(config)
	if err != nil {
		panic("Failed to connect to test database: " + err.Error())
	}

	// Run tests
	code := m.Run()

	// Cleanup
	testDB.Close()
	os.Exit(code)
}

func setupTestData(t *testing.T) *TodoRepository {
	// Clean the database
	_, err := testDB.Exec("TRUNCATE todos RESTART IDENTITY CASCADE")
	require.NoError(t, err)

	return NewTodoRepository(testDB)
}

func TestTodoRepository_Create(t *testing.T) {
	repo := setupTestData(t)

	req := models.TodoRequest{
		Title:       "Test Todo",
		Description: "Test Description",
		Completed:   false,
	}

	todo, err := repo.Create(req)

	assert.NoError(t, err)
	assert.NotZero(t, todo.ID)
	assert.Equal(t, req.Title, todo.Title)
	assert.Equal(t, req.Description, todo.Description)
	assert.Equal(t, req.Completed, todo.Completed)
	assert.NotZero(t, todo.CreatedAt)
	assert.NotZero(t, todo.UpdatedAt)
}

func TestTodoRepository_GetByID(t *testing.T) {
	repo := setupTestData(t)

	// Create a todo first
	req := models.TodoRequest{
		Title:       "Test Todo",
		Description: "Test Description",
		Completed:   false,
	}

	created, err := repo.Create(req)
	require.NoError(t, err)

	// Get the todo
	found, err := repo.GetByID(created.ID)

	assert.NoError(t, err)
	assert.Equal(t, created.ID, found.ID)
	assert.Equal(t, created.Title, found.Title)
}

func TestTodoRepository_GetAll_WithFilters(t *testing.T) {
	repo := setupTestData(t)

	// Create test data
	todos := []models.TodoRequest{
		{Title: "Learn Go", Description: "Study basics", Completed: false},
		{Title: "Build API", Description: "REST API", Completed: true},
		{Title: "Learn Python", Description: "Study advanced", Completed: false},
	}

	for _, todo := range todos {
		_, err := repo.Create(todo)
		require.NoError(t, err)
	}

	// Test filtering by completion status
	filter := models.TodoFilter{
		Completed: &[]bool{false}[0],
		Limit:     10,
		Offset:    0,
	}

	results, err := repo.GetAll(filter)
	assert.NoError(t, err)
	assert.Len(t, results, 2)

	// Test search functionality
	filter = models.TodoFilter{
		Search: "Learn",
		Limit:  10,
		Offset: 0,
	}

	results, err = repo.GetAll(filter)
	assert.NoError(t, err)
	assert.Len(t, results, 2)
}

func TestTodoRepository_Update(t *testing.T) {
	repo := setupTestData(t)

	// Create a todo
	req := models.TodoRequest{
		Title:       "Original Title",
		Description: "Original Description",
		Completed:   false,
	}

	created, err := repo.Create(req)
	require.NoError(t, err)

	// Update the todo
	updateReq := models.TodoRequest{
		Title:       "Updated Title",
		Description: "Updated Description",
		Completed:   true,
	}

	updated, err := repo.Update(created.ID, updateReq)

	assert.NoError(t, err)
	assert.Equal(t, created.ID, updated.ID)
	assert.Equal(t, updateReq.Title, updated.Title)
	assert.Equal(t, updateReq.Description, updated.Description)
	assert.Equal(t, updateReq.Completed, updated.Completed)
	assert.True(t, updated.UpdatedAt.After(created.UpdatedAt))
}

func TestTodoRepository_Delete(t *testing.T) {
	repo := setupTestData(t)

	// Create a todo
	req := models.TodoRequest{
		Title:       "To Delete",
		Description: "Will be deleted",
		Completed:   false,
	}

	created, err := repo.Create(req)
	require.NoError(t, err)

	// Delete the todo
	err = repo.Delete(created.ID)
	assert.NoError(t, err)

	// Verify it's deleted
	_, err = repo.GetByID(created.ID)
	assert.Error(t, err)
}
