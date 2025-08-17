"""
OCR Version Switcher - Compare Optimized vs Standard Performance
"""

import tkinter as tk
from tkinter import ttk
import time
from versions.ocr_standard import StandardDigitDrawGUI
from versions.ocr_optimized import OptimizedDigitDrawGUI


class OCRVersionSwitcher:
    """Main controller that switches between Optimized and Standard versions"""
    
    def __init__(self, model):
        self.model = model
        self.current_version = "optimized"
        self.performance_stats = {"standard": [], "optimized": []}
        
        # Initialize both versions
        self.standard_ocr = StandardDigitDrawGUI(model, show_ui=False)
        self.optimized_ocr = OptimizedDigitDrawGUI(model, show_ui=False)
        
        # Create main UI
        self.setup_main_ui()
        
        # Start with optimized version
        self.current_ocr = self.optimized_ocr
        self.setup_current_version()
        
    def setup_main_ui(self):
        """Setup the main UI with version switcher and stats area"""
        self.root = tk.Tk()
        self.root.title("OCR Performance Comparison - Optimized vs Standard")
        self.root.geometry("1200x800")  # Set initial size to accommodate stats panel
        
        # Main container - use grid for better layout control
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=2)  # OCR area gets more space
        main_frame.grid_columnconfigure(1, weight=1)  # Stats area
        main_frame.grid_rowconfigure(1, weight=1)     # Content area expands
        
        # Version control panel (spans both columns)
        control_panel = ttk.LabelFrame(main_frame, text="Version Control & Performance", padding=10)
        control_panel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # Version switcher
        version_frame = ttk.Frame(control_panel)
        version_frame.pack(fill="x", pady=(0, 5))
        
        ttk.Label(version_frame, text="Current Version:").pack(side="left")
        
        self.version_var = tk.StringVar(value="Optimized")
        self.version_label = ttk.Label(version_frame, textvariable=self.version_var, 
                                      font=("Helvetica", 12, "bold"), foreground="green")
        self.version_label.pack(side="left", padx=(10, 20))
        
        self.switch_btn = ttk.Button(version_frame, text="Switch to Standard", 
                                   command=self.switch_version)
        self.switch_btn.pack(side="left", padx=(0, 20))
        
        # Performance display
        perf_frame = ttk.Frame(control_panel)
        perf_frame.pack(fill="x")
        
        ttk.Label(perf_frame, text="Last Prediction Time:").pack(side="left")
        self.time_var = tk.StringVar(value="— ms")
        self.time_label = ttk.Label(perf_frame, textvariable=self.time_var, 
                                  font=("Helvetica", 11, "bold"))
        self.time_label.pack(side="left", padx=(10, 20))
        
        ttk.Label(perf_frame, text="Avg Time (last 10):").pack(side="left")
        self.avg_time_var = tk.StringVar(value="— ms")
        self.avg_time_label = ttk.Label(perf_frame, textvariable=self.avg_time_var, 
                                      font=("Helvetica", 11, "bold"))
        self.avg_time_label.pack(side="left", padx=(10, 20))
        
        # Hide stats button (now just toggles detailed view)
        self.stats_btn = ttk.Button(perf_frame, text="Hide Detailed Stats", 
                                  command=self.toggle_performance_stats)
        self.stats_btn.pack(side="right")
        
        # Content area with two columns
        # Left column: OCR interface container
        self.ocr_container = ttk.Frame(main_frame)
        self.ocr_container.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        
        # Right column: Performance stats area 
        self.stats_frame = ttk.LabelFrame(main_frame, text="Performance Statistics", padding=10)
        self.stats_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self.stats_visible = True  # Stats are now visible by default
        
        # Create stats text widget with scrollbar
        stats_text_frame = ttk.Frame(self.stats_frame)
        stats_text_frame.pack(fill="both", expand=True)
        
        self.stats_text = tk.Text(stats_text_frame, wrap="word", font=("Consolas", 10),
                                 width=40, height=20, state="disabled")
        stats_scrollbar = ttk.Scrollbar(stats_text_frame, orient="vertical", 
                                       command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side="left", fill="both", expand=True)
        stats_scrollbar.pack(side="right", fill="y")
        
        # Auto-refresh checkbox for stats
        self.auto_refresh_var = tk.BooleanVar(value=True)
        auto_refresh_cb = ttk.Checkbutton(self.stats_frame, text="Auto-refresh stats", 
                                         variable=self.auto_refresh_var)
        auto_refresh_cb.pack(pady=(5, 0), anchor="w")
        
        # Refresh button
        refresh_btn = ttk.Button(self.stats_frame, text="Refresh Stats", 
                               command=self.refresh_performance_stats)
        refresh_btn.pack(pady=(5, 0), anchor="w")
        
        # Initialize stats display
        self.refresh_performance_stats()
        
    def setup_current_version(self):
        """Setup the current OCR version in the container"""
        # Clear container
        for widget in self.ocr_container.winfo_children():
            widget.destroy()
            
        # Move current OCR's UI to our container
        if hasattr(self.current_ocr, 'main_frame'):
            self.current_ocr.main_frame.pack_forget()
            
        # Create the OCR interface in our container
        self.current_ocr.setup_ui_in_container(self.ocr_container)
        
        # Hook into the prediction method to measure performance
        original_predict = self.current_ocr.predict_and_display
        
        def timed_predict(*args, **kwargs):
            start_time = time.time()
            result = original_predict(*args, **kwargs)
            end_time = time.time()
            
            prediction_time = (end_time - start_time) * 1000  # Convert to ms
            self.update_performance_stats(prediction_time)
            return result
            
        self.current_ocr.predict_and_display = timed_predict
        
    def switch_version(self):
        """Switch between standard and optimized versions"""
        self.current_ocr.clear_canvas()

        if self.current_version == "standard":
            self.current_version = "optimized"
            self.current_ocr = self.optimized_ocr
            self.version_var.set("Optimized")
            self.version_label.configure(foreground="green")
            self.switch_btn.config(text="Switch to Standard")
        else:
            self.current_version = "standard" 
            self.current_ocr = self.standard_ocr
            self.version_var.set("Standard")
            self.version_label.configure(foreground="red")
            self.switch_btn.config(text="Switch to Optimized")

        self.setup_current_version()
        
    def update_performance_stats(self, prediction_time):
        """Update performance statistics"""
        stats = self.performance_stats[self.current_version]
        stats.append(prediction_time)
        
        # Keep only last 50 measurements
        if len(stats) > 50:
            stats.pop(0)
            
        # Update display
        self.time_var.set(f"{prediction_time:.1f} ms")
        
        if len(stats) >= 2:
            recent_avg = sum(stats[-10:]) / min(len(stats), 10)
            self.avg_time_var.set(f"{recent_avg:.1f} ms")
            
        # Color coding based on performance
        if prediction_time < 100:
            self.time_label.config(foreground="green")
        elif prediction_time < 300:
            self.time_label.config(foreground="orange")
        else:
            self.time_label.config(foreground="red")
            
        # Auto-refresh stats if enabled
        if self.auto_refresh_var.get():
            self.refresh_performance_stats()
            
    def toggle_performance_stats(self):
        """Toggle the visibility of the performance stats panel"""
        if self.stats_visible:
            # Hide stats panel
            self.stats_frame.grid_remove()
            self.stats_btn.config(text="Show Detailed Stats")
            self.stats_visible = False
            
            # Adjust main frame grid to give OCR container full width
            self.root.geometry("")  # Reset to auto-size
            
        else:
            # Show stats panel
            self.stats_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
            self.stats_btn.config(text="Hide Detailed Stats")
            self.stats_visible = True
            
            # Refresh stats content
            self.refresh_performance_stats()
            
            # Adjust window size to accommodate stats panel
            self.root.update_idletasks()  # Ensure geometry is calculated
            current_width = self.root.winfo_width()
            if current_width < 1200:  # Expand window if too narrow
                self.root.geometry("1200x800")
    
    def refresh_performance_stats(self):
        """Update the stats text widget with current performance data"""
        if not self.stats_visible:
            return
            
        stats_text = self.generate_stats_report()
        
        # Update text widget
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", stats_text)
        self.stats_text.config(state="disabled")
        
        # Scroll to top
        self.stats_text.see("1.0")
        
    def generate_stats_report(self):
        """Generate detailed performance report"""
        report = "=== OCR Performance Comparison ===\n\n"

        report += f"Last updated: {time.strftime('%H:%M:%S')}\n\n"
        
        
        for version in ["optimized", "standard"]:
            stats = self.performance_stats[version]
            if not stats:
                report += f"{version.upper()} VERSION:\n"
                report += "  No data available\n\n"
                continue
                
            avg_time = sum(stats) / len(stats)
            min_time = min(stats)
            max_time = max(stats)
            
            # Calculate percentiles for better insight
            sorted_stats = sorted(stats)
            n = len(sorted_stats)
            p50 = sorted_stats[n//2] if n > 0 else 0
            p95 = sorted_stats[int(n*0.95)] if n > 0 else 0
            
            report += f"{version.upper()} VERSION:\n"
            report += f"  Average: {avg_time:.1f} ms\n"
            report += f"  Minimum: {min_time:.1f} ms\n" 
            report += f"  Maximum: {max_time:.1f} ms\n"
            report += f"  Median (50th): {p50:.1f} ms\n"
            report += f"  95th percentile: {p95:.1f} ms\n"
            report += f"  Last 5: {[f'{t:.1f}' for t in stats[-5:]]}\n"
            
            # Performance trend (last 10 vs first 10)
            if len(stats) >= 20:
                first_10_avg = sum(stats[:10]) / 10
                last_10_avg = sum(stats[-10:]) / 10
                trend = "improving" if last_10_avg < first_10_avg else "declining"
                trend_diff = abs(last_10_avg - first_10_avg)
                report += f"  Trend: {trend} ({trend_diff:.1f} ms)\n"
                
            report += "\n"
            
        # Comparison
        std_stats = self.performance_stats["standard"]
        opt_stats = self.performance_stats["optimized"]
        
        if std_stats and opt_stats:
            std_avg = sum(std_stats) / len(std_stats)
            opt_avg = sum(opt_stats) / len(opt_stats)
            speedup = std_avg / opt_avg if opt_avg > 0 else 0
            
            report += "COMPARISON:\n"
            if speedup > 1:
                report += f"  🚀 Optimized is {speedup:.2f}x faster\n"
                report += f"  💾 Time saved: {std_avg - opt_avg:.1f} ms per prediction\n"
            elif speedup < 1:
                report += f"  📊 Standard is {1/speedup:.2f}x faster\n"
                report += f"  💾 Time penalty: {opt_avg - std_avg:.1f} ms per prediction\n"
            else:
                report += f"  ⚖️ Performance is similar\n"
                
            # Estimate hourly savings
            if len(std_stats) > 0 and len(opt_stats) > 0:
                predictions_per_hour = 3600 / ((std_avg + opt_avg) / 2000)  # Rough estimate
                hourly_savings = abs(std_avg - opt_avg) * predictions_per_hour / 1000
                report += f"  ⏱️ Est. hourly savings: {hourly_savings:.1f} seconds\n"
            
            report += "\n"
            
        report += "NOTES:\n"
        report += "• Times include UI updates and drawing\n"
        report += "• Pure ML inference may be faster\n"
        report += "• Draw multiple digits for bigger differences\n"
        
        return report
        
    def run(self):
        """Run the application"""
        self.root.mainloop()