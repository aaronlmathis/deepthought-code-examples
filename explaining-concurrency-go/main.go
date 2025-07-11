package main

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"
)

var urls = []string{
	"https://www.deepthought.sh",
	"https://www.google.com",
	"https://www.github.com",
	"https://www.stackoverflow.com",
	"https://www.reddit.com",
	"https://www.wikipedia.org",
	"https://www.youtube.com",
	"https://www.twitter.com",
	"https://www.facebook.com",
	"https://www.amazon.com",
	"https://www.microsoft.com",
	"https://www.apple.com",
	"https://www.linkedin.com",
	"https://www.instagram.com",
	"https://www.netflix.com",
	"https://www.nytimes.com",
	"https://www.cnn.com",
	"https://www.bbc.com",
	"https://www.medium.com",
	"https://www.dropbox.com",
	"https://www.spotify.com",
	"https://www.salesforce.com",
	"https://www.slack.com",
	"https://www.airbnb.com",
	"https://www.udemy.com",
	"https://www.khanacademy.org",
	"https://www.quora.com",
	"https://www.ted.com",
	"https://www.nationalgeographic.com",
	"https://www.imdb.com",
}

type Result struct {
	URL      string
	Status   int
	Error    error
	Duration time.Duration
}

func fetchURL(ctx context.Context, url string, results chan<- Result, wg *sync.WaitGroup) {
	defer wg.Done()

	start := time.Now()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		results <- Result{
			URL:      url,
			Status:   0,
			Error:    err,
			Duration: time.Since(start),
		}
		return
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	duration := time.Since(start)

	if err != nil {
		results <- Result{
			URL:      url,
			Status:   0,
			Error:    err,
			Duration: duration,
		}
		return
	}
	defer resp.Body.Close()

	results <- Result{
		URL:      url,
		Status:   resp.StatusCode,
		Error:    nil,
		Duration: duration,
	}
}

func workerPool(urls []string, numWorkers int, results chan Result) {
	jobs := make(chan string, len(urls))

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker(jobs, results, &wg)
	}

	// Send jobs
	for _, url := range urls {
		jobs <- url
	}
	close(jobs)

	// Wait for workers to finish
	wg.Wait()
}

func worker(jobs <-chan string, results chan<- Result, wg *sync.WaitGroup) {
	defer wg.Done()

	for url := range jobs {
		start := time.Now()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err != nil {
			results <- Result{URL: url, Error: err, Duration: time.Since(start)}
			continue
		}

		client := &http.Client{}
		resp, err := client.Do(req)
		duration := time.Since(start)

		if err != nil {
			results <- Result{URL: url, Error: err, Duration: duration}
			continue
		}
		resp.Body.Close()

		results <- Result{
			URL:      url,
			Status:   resp.StatusCode,
			Duration: duration,
		}
	}
}

func processResults(results <-chan Result) {
	var successful, failed []Result
	var totalDuration time.Duration

	for result := range results {
		if result.Error != nil {
			failed = append(failed, result)
		} else {
			successful = append(successful, result)
			totalDuration += result.Duration // Sum of individual request durations
		}
	}

	// Print detailed results
	fmt.Println("\n=== SUCCESSFUL REQUESTS ===")
	for _, result := range successful {
		fmt.Printf("[HEALTHY] %s - Status: %d (took %v)\n",
			result.URL, result.Status, result.Duration)
	}

	if len(failed) > 0 {
		fmt.Println("\n=== FAILED REQUESTS ===")
		for _, result := range failed {
			fmt.Printf("[UNHEALTHY] %s - Error: %v (took %v)\n",
				result.URL, result.Error, result.Duration)
		}
	}

	// Print summary
	fmt.Printf("\n=== SUMMARY ===\n")
	fmt.Printf("Total URLs: %d\n", len(successful)+len(failed))
	fmt.Printf("Successful: %d\n", len(successful))
	fmt.Printf("Failed: %d\n", len(failed))
	fmt.Printf("Success rate: %.1f%%\n", float64(len(successful))/float64(len(successful)+len(failed))*100)
	fmt.Printf("Total time (sum of request durations): %v\n", totalDuration)
	if len(successful) > 0 {
		fmt.Printf("Average response time: %v\n", totalDuration/time.Duration(len(successful)))
	}
}

func main() {
	fmt.Println("Starting web scraper with worker pool and summary...")
	start := time.Now()

	numWorkers := 5 // Limit concurrency to 5 workers
	results := make(chan Result, len(urls))

	workerPool(urls, numWorkers, results)
	close(results) // Close the results channel after workerPool completes
	processResults(results)

	// Print the actual elapsed time
	fmt.Printf("\nCompleted in %v\n", time.Since(start))
}
