import os
import time
import logging
import csv
import psutil
import threading
import json
from datetime import datetime

log = logging.getLogger(__name__)


class ResourceMonitor:
    """Real-time resource monitoring for solver processes"""
    
    def __init__(self, monitor_output_dir=None):
        self.monitor_output_dir = monitor_output_dir
        self.monitoring = False
        self.monitor_thread = None
        self.process_data = []
        self._process_objs = {}  # Persist Process objects per PID
        
        if self.monitor_output_dir:
            os.makedirs(self.monitor_output_dir, exist_ok=True)
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        log.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring and save data"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        if self.monitor_output_dir and self.process_data:
            self._save_monitoring_data()
        
        log.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Monitor all running solver processes
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'cmdline']):
                    try:
                        if proc.info['name'] in ['z3', 'z3-4.8.8'] or any('z3' in str(arg) for arg in proc.info['cmdline'] or []):
                            pid = proc.info['pid']
                            # Reuse Process object if possible
                            if pid not in self._process_objs:
                                self._process_objs[pid] = psutil.Process(pid)
                            process = self._process_objs[pid]
                            cpu_percent = process.cpu_percent(interval=None)  # Use None for non-blocking
                            memory_mb = process.memory_info().rss / (1024 * 1024)
                            
                            try:
                                cpu_affinity = process.cpu_affinity()
                            except (psutil.AccessDenied, AttributeError):
                                cpu_affinity = []
                            
                            process_info = {
                                'timestamp': current_time,
                                'pid': pid,
                                'cpu_percent': cpu_percent,
                                'memory_mb': round(memory_mb, 2),
                                'cpu_affinity': cpu_affinity,
                                'num_threads': process.num_threads(),
                                'status': process.status()
                            }
                            
                            self.process_data.append(process_info)
                            
                            # Log high resource usage
                            if cpu_percent > 90:
                                log.warning(f"High CPU usage: PID {pid} using {cpu_percent:.1f}% CPU")
                            if memory_mb > 500:  # > 500MB
                                log.warning(f"High memory usage: PID {pid} using {memory_mb:.1f} MB")
                    
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                
            except Exception as e:
                log.error(f"Error in monitoring loop: {e}")
            
            time.sleep(3)  # Monitor every 3 seconds
    
    def _save_monitoring_data(self):
        """Save collected monitoring data"""
        if not self.process_data:
            return
        
        # Save as CSV
        csv_file = os.path.join(self.monitor_output_dir, "detailed_process_monitor.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'pid', 'cpu_percent', 'memory_mb', 
                                                  'cpu_affinity', 'num_threads', 'status'])
            writer.writeheader()
            writer.writerows(self.process_data)
        
        # Save as JSON for detailed analysis
        json_file = os.path.join(self.monitor_output_dir, "process_monitor.json")
        with open(json_file, 'w') as f:
            json.dump(self.process_data, f, indent=2, default=str)
        
        # Generate summary statistics
        self._generate_summary()
        
        log.info(f"Monitoring data saved to {self.monitor_output_dir}")
    
    def _generate_summary(self):
        """Generate monitoring summary statistics"""
        if not self.process_data:
            return
        
        summary_file = os.path.join(self.monitor_output_dir, "monitoring_summary.txt")
        
        # Calculate statistics
        cpu_values = [d['cpu_percent'] for d in self.process_data if d['cpu_percent'] > 0]
        memory_values = [d['memory_mb'] for d in self.process_data]
        unique_pids = set(d['pid'] for d in self.process_data)
        
        with open(summary_file, 'w') as f:
            f.write("RESOURCE MONITORING SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Monitoring duration: {len(self.process_data)} data points\n")
            f.write(f"Unique processes monitored: {len(unique_pids)}\n")
            f.write(f"PIDs: {sorted(unique_pids)}\n\n")
            
            if cpu_values:
                f.write(f"CPU Usage Statistics:\n")
                f.write(f"  Average: {sum(cpu_values)/len(cpu_values):.2f}%\n")
                f.write(f"  Peak: {max(cpu_values):.2f}%\n")
                f.write(f"  Min: {min(cpu_values):.2f}%\n\n")
            
            if memory_values:
                f.write(f"Memory Usage Statistics:\n")
                f.write(f"  Average: {sum(memory_values)/len(memory_values):.2f} MB\n")
                f.write(f"  Peak: {max(memory_values):.2f} MB\n")
                f.write(f"  Min: {min(memory_values):.2f} MB\n\n")
            
            # CPU affinity analysis
            affinity_data = [d['cpu_affinity'] for d in self.process_data if d['cpu_affinity']]
            if affinity_data:
                f.write(f"CPU Affinity Analysis:\n")
                unique_affinities = set(tuple(sorted(aff)) for aff in affinity_data)
                for aff in unique_affinities:
                    count = sum(1 for a in affinity_data if tuple(sorted(a)) == aff)
                    f.write(f"  CPUs {aff}: {count} observations\n")


def log_resource_usage(pid, task_id, duration):
    """Log resource usage for a specific process"""
    try:
        process = psutil.Process(pid)
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                cpu_percent = process.cpu_percent(interval=1)
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_affinity = process.cpu_affinity()
                
                log.debug(f"Task {task_id} (PID {pid}): CPU={cpu_percent:.1f}%, "
                         f"Memory={memory_mb:.1f}MB, Affinity={cpu_affinity}")
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            
            time.sleep(5)  # Log every 5 seconds
            
    except Exception as e:
        log.debug(f"Resource logging failed for task {task_id}: {e}")