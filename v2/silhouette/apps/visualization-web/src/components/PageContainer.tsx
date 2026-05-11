import type { PropsWithChildren } from "react";

import { motion } from "framer-motion";

type PageContainerProps = PropsWithChildren<{
  title: string;
  description: string;
}>;

export function PageContainer({ title, description, children }: PageContainerProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="page-container"
    >
      <header className="page-header">
        <h1>{title}</h1>
        <p>{description}</p>
      </header>
      {children}
    </motion.div>
  );
}
