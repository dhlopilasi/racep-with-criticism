### 1. Tạo session và chạy experiment

```bash
cd ~/Documents/Classroom/KLTN/protree
source .venv-racep/bin/activate && conda deactivate

tmux new -s racep
python experiments/run_racep_sine.py > logs/racep_sine_run001.log 2>&1
```

Ở đây `run001` đổi thành `run002`, `run003`... cho mỗi lần chạy mới.

### 2. Rời session nhưng để Python tiếp tục chạy

Nhấn:

```text
Ctrl+B
```

rồi nhấn:

```text
D
```

### 3. Xem các tmux session đang chạy

```bash
tmux ls
```

Ví dụ:

```text
racep: 1 windows
```

### 4. Quay lại session đang chạy

```bash
tmux attach -t racep
```

### 5. Xem log mà không cần attach tmux

```bash
tail -f logs/racep_sine_run001.log
```

Thoát `tail -f` bằng:

```text
Ctrl+C
```

**Không làm dừng experiment**; nó chỉ dừng việc xem log.

### 6. Khi experiment đã chạy xong

Có thể kiểm tra:

```bash
cat logs/racep_sine_run001.log
```

hoặc xóa session:

```bash
tmux kill-session -t racep
```

### Note ngắn gọn nhất để em lưu

```text
TMUX WORKFLOW

cd ~/Documents/Classroom/KLTN/protree
source .venv-racep/bin/activate

tmux new -s racep
python experiments/run_xxx.py > logs/run001.log 2>&1

Detach: Ctrl+B → D
Xem session: tmux ls
Vào lại: tmux attach -t racep
Xem log: tail -f logs/run001.log
```

Một lưu ý: **mỗi lần chạy nên đổi tên log** (`run001`, `run002`, …), để không ghi đè kết quả cũ.
