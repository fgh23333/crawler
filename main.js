const { Worker } = require('worker_threads');
const numThreads = 256;
const totalTasks = 1000;
let completedTasks = 0;

for (let i = 0; i < numThreads; i++) {
  const worker = new Worker('./new_index.js');
  worker.on('message', (result) => {
    if (result) {
        console.log(`${result}`);
        completedTasks++;
    } else {
        completedTasks++;
    }
    if (completedTasks < totalTasks) {
      worker.postMessage(completedTasks + numThreads);
    }
  });
  worker.postMessage(i + 1);
}
