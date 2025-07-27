package database

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config holds the database configuration settings
type Config struct {
	Host            string
	Port            int
	User            string
	Password        string
	DBName          string
	SSLMode         string
	MaxOpenConns    int
	MaxIdleConns    int
	ConnMaxLifetime time.Duration
	ConnMaxIdleTime time.Duration
}

// LoadConfig loads the database configuration from environment variables
func LoadConfig() Config {
	return Config{
		Host:            getEnvWithDefault("DB_HOST", "localhost"),
		Port:            getEnvAsIntWithDefault("DB_PORT", 5432),
		User:            getEnvWithDefault("DB_USER", "todouser"),
		Password:        getEnvWithDefault("DB_PASSWORD", "todopass"),
		DBName:          getEnvWithDefault("DB_NAME", "todoapi"),
		SSLMode:         getEnvWithDefault("DB_SSLMODE", "disable"),
		MaxOpenConns:    getEnvAsIntWithDefault("DB_MAX_OPEN_CONNS", 25),
		MaxIdleConns:    getEnvAsIntWithDefault("DB_MAX_IDLE_CONNS", 5),
		ConnMaxLifetime: getEnvAsDurationWithDefault("DB_CONN_MAX_LIFETIME", 5*time.Minute),
		ConnMaxIdleTime: getEnvAsDurationWithDefault("DB_CONN_MAX_IDLE_TIME", 5*time.Minute),
	}
}

// ConnectionString returns the database connection string
func (c Config) ConnectionString() string {
	return fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		c.Host, c.Port, c.User, c.Password, c.DBName, c.SSLMode)
}

// getEnvWithDefault retrieves an environment variable with a default value
func getEnvWithDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getEnvAsIntWithDefault retrieves an environment variable as an integer with a default value
func getEnvAsIntWithDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

// getEnvAsDurationWithDefault retrieves an environment variable as a time.Duration with a default value
func getEnvAsDurationWithDefault(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}
