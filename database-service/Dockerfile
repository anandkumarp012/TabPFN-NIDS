
FROM node:22-slim

WORKDIR /app

# Use the same npm major version as the local environment
RUN npm install -g npm@11.1.0

COPY package.json package-lock.json ./
RUN npm install --no-audit --no-fund

COPY prisma ./prisma
COPY prisma.config.ts ./
COPY src ./src

RUN DIRECT_URL="postgresql://user:password@localhost:5432/placeholder" npm run db:generate

EXPOSE 3000

CMD ["npm", "start"]
