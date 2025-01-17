install_github("wjawaid/enrichR")
install.packages("enrichR")
install.packages("ggplot2")
install.packages("ggrepel")

library(devtools)
library(enrichR)
library(ggplot2)
library(dplyr)
library(stringr)
library(ggrepel)
library(tidyverse)
library(readxl)

listEnrichrSites()
setEnrichrSite("Enrichr")
dbs <- listEnrichrDbs()
dbs <- c("Reactome_2022", "KEGG_2021_Human", "GO_Biological_Process_2023")

# wases <- c('GWAS', 'TWAS', 'PWAS')

wases = c('TWAS')
for (was in wases) {
  print(was)
  annot <- read.csv("~/Desktop/Thesis_Work/Projects/AD_WAS/Circos/AD_annotations_fin.tsv", sep = '\t')
  hits <- annot[annot$group %in% was,]
  print(hits)
  
  enriched <- enrichr(hits$label, dbs)
  
  df1 <- data.frame(
    Pathway = enriched$KEGG_2021_Human$Term,
    adjpvalue = enriched$KEGG_2021_Human$Adjusted.P.value,
    pvalue = enriched$KEGG_2021_Human$P.value,
    GeneCount = enriched$KEGG_2021_Human$Genes,
    CombinedScore = enriched$KEGG_2021_Human$Combined.Score
  )
  
  df2 <- data.frame(
    Pathway = enriched$Reactome_2022$Term,
    adjpvalue = enriched$Reactome_2022$Adjusted.P.value,
    pvalue = enriched$Reactome_2022$P.value,
    GeneCount = enriched$Reactome_2022$Genes,
    CombinedScore = enriched$Reactome_2022$Combined.Score
  )
  
  df3 <- data.frame(
    Pathway = enriched$GO_Biological_Process_2023$Term,
    adjpvalue = enriched$GO_Biological_Process_2023$Adjusted.P.value,
    pvalue = enriched$GO_Biological_Process_2023$P.value,
    GeneCount = enriched$GO_Biological_Process_2023$Genes,
    CombinedScore = enriched$GO_Biological_Process_2023$Combined.Score
  )
  
  
  # Add the Source column with a repeated string
  df1 <- df1 %>%
    mutate(Source = "Kegg_2021_Human")
  df2 <- df2 %>%
    mutate(Source = "Reactome_2022")
  df3 <- df3 %>%
    mutate(Source = "GO_Biological_Process_2023")
  
  # Combine data frames
  combined_df <- bind_rows(df1, df2, df3)
  
  # Calculate the count of genes per row
  combined_df <- combined_df %>%
    mutate(GeneCountCount = str_count(GeneCount, ";") + 1)
  
  # Calculate log10 p-values
  combined_df <- combined_df %>%
    mutate(log10_pvalue = -log10(pvalue))
  
  filtered_df <- combined_df %>% filter(adjpvalue < 1)
  # filtered_df <- filtered_df %>% filter(CombinedScore > 90)
  
  filt_file <- sprintf("~/Desktop/Thesis_Work/Projects/AD_WAS/enrichr/%s_filtered_pathways_kegg_reactome.csv", was)
  write.csv(filtered_df, filt_file)
  
  plot <- ggplot(head(filtered_df, 20), aes(x = CombinedScore, y = reorder(Pathway, log10_pvalue), 
                                            size = GeneCountCount, color = log10_pvalue, shape = Source)) +
    geom_point() +
    scale_color_gradient(low = "blue", high = "red") +
    labs(
      x = "CombinedScore",
      y = "Pathway",
      size = "Gene Count",
      color = expression(-log[10](p-value)),
      shape = "Source",
      title = sprintf("%s Pathway Enrichment Analysis", was)
    ) +
    theme_minimal()
  
  print('done')
  filename <- sprintf("~/Desktop/Thesis_Work/Projects/AD_WAS/enrichr/%s.png", was)
  ggsave(filename = filename, plot = plot, width = 10, height = 8, units = "in", dpi = 300)
  
  
}
