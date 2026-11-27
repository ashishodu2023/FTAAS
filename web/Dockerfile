# Product site only. Build from the repository root:
#   docker build -f web/Dockerfile -t ftaas-web .
# The Python gateway image stays Dockerfile at the repo root.
FROM node:22-alpine
WORKDIR /app
COPY web/package.json web/server.js web/layout.js web/pages.js web/catalog-data.js ./
COPY web/public ./public
EXPOSE 3000
# Host-assigned PORT wins at runtime (see server.js).
CMD ["node", "server.js"]
