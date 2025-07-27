package handlers

import (
	"net/http"
	"strconv"
	"strings"

	"todo-api/internal/models"
	"todo-api/internal/repository"

	"github.com/gin-gonic/gin"
)

type TodoHandler struct {
	repo *repository.TodoRepository
}

func NewTodoHandler(repo *repository.TodoRepository) *TodoHandler {
	return &TodoHandler{
		repo: repo,
	}
}

func (h *TodoHandler) GetTodos(c *gin.Context) {
	var filter models.TodoFilter
	if err := c.ShouldBindQuery(&filter); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid query parameters",
			"details": err.Error(),
		})
		return
	}

	todos, err := h.repo.GetAll(filter)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to retrieve todos",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"todos":  todos,
		"count":  len(todos),
		"filter": filter,
	})
}

func (h *TodoHandler) GetTodo(c *gin.Context) {
	idParam := c.Param("id")
	id, err := strconv.Atoi(idParam)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid todo ID format",
		})
		return
	}

	todo, err := h.repo.GetByID(id)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "Todo not found",
			})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to retrieve todo",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"todo": todo,
	})
}

func (h *TodoHandler) CreateTodo(c *gin.Context) {
	var req models.TodoRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request data",
			"details": err.Error(),
		})
		return
	}

	todo, err := h.repo.Create(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to create todo",
		})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"todo":    todo,
		"message": "Todo created successfully",
	})
}

func (h *TodoHandler) UpdateTodo(c *gin.Context) {
	idParam := c.Param("id")
	id, err := strconv.Atoi(idParam)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid todo ID format",
		})
		return
	}

	var req models.TodoRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request data",
			"details": err.Error(),
		})
		return
	}

	todo, err := h.repo.Update(id, req)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "Todo not found",
			})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to update todo",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"todo":    todo,
		"message": "Todo updated successfully",
	})
}

func (h *TodoHandler) DeleteTodo(c *gin.Context) {
	idParam := c.Param("id")
	id, err := strconv.Atoi(idParam)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid todo ID format",
		})
		return
	}

	err = h.repo.Delete(id)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "Todo not found",
			})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to delete todo",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "Todo deleted successfully",
	})
}

func (h *TodoHandler) GetStats(c *gin.Context) {
	// For now, return basic stats - this can be enhanced later
	c.JSON(http.StatusOK, gin.H{
		"message": "Stats endpoint - implementation pending",
		"status":  "success",
	})
}
