/**
 * Start the product site on a non-default port (Windows-safe; no shell env syntax).
 * Usage: node run-port.js [port]
 */
process.env.PORT = String(process.argv[2] || "3001");
await import("./server.js");
