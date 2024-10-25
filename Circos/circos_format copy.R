# Functions and code that will run to format the input files to be ready for the 
# make_circos.CMVD.rmd

rm(list = ls())

################################################################################
#                                 functions                                    #
################################################################################

#' Formats data for gene input
#' @param data data.frame containing biofilter output

#data = genes_gr_c
format_genes <- function(data){
  
  data <- as.data.frame(data)
  data = data[order(data$P),]
  data2 <- data[,c(2,3,3,16,24)] # label = search gene (not gene)
  data2$POS.1 <- data2$POS.1 + 1
  names(data2)[1:5] <- c("chr", "start", "end", "label","group") 
  data2$chr <- paste0("chr", data2$chr)
  data2$start <- as.numeric(data2$start)
  data3 = unique(data2)
  data3 = data3[!duplicated(data3$label), ]
  
  data.bed <- data3[data3$chr != "chrNA", ]
  
  return(data.bed)
}



#' #' Formats data for heterozygosity track
#' #' @param data data.frame containing meta analysis output
#' format_het <- function(data){
#'   
#'   data <- as.data.frame(data)
#'   
#'   data$end <- data$BP + 1
#'   
#'   data.het <- data[, c("CHR", "BP", "end", "I")]
#'   
#'   names(data.het) <- c("chr", "start", "end", "value1")
#'   
#'   data.het$chr <- paste0("chr", data.het$chr)
#'   
#'   return(data.het)
#' }

#' Formats data for Manhattan plot
#' @param data data.frame containing meta analysis output
#' @param annotation data.frame containing annotations
#' @param p.threshold p-value to filter out

#data = mega.data2 # for testing
#annotations = anno.data2 # for testing
format_manhattan <- function (data, annotations, 
                             p.threshold,
                             col = "black",
                             anno.col = "#009E73"){
  
  
  data <- as.data.frame(data)
  annotations <- as.data.frame(annotations)
  # data$BP <- data$position
  data$CHR <- data$CHROM
  data$BP <- data$POS
  data$end <- data$BP + 1
  data$value1 <- -log10(data$P)
  data$value2 <- col
  
  data.fmt <- data[data$value1 > -log10(p.threshold) | data$value2 == 1,
                   c("CHR", "POS", "end", "value1", "value2")]
  
  names(data.fmt)[1:2] <- c("chr", "start")
  
  
  data.fmt$chr <- paste0("chr", data.fmt$chr)
  
  return(data.fmt)  
}

################################################################################
#                                 file paths                                   #
################################################################################

# inputs 
# Four input files here
gwas.path <- file.path("~/Desktop/Thesis_Work/Projects/AD_WAS/Circos/ADSP_keep_quest_comb_cohorts_GWAS_age_sex_PC1-5_fin.ALZ_STATUS.glm.logistic.hybrid")
twas.path <- file.path("~/Desktop/Thesis_Work/Projects/AD_WAS/Circos/ADSP_TWAS_GTEx.all_tissues_pos_fin.tsv")
pwas.path <- file.path("~/Desktop/Thesis_Work/Projects/AD_WAS/Circos/ADSP_PWAS_ARIC.all_tissues_pos_fin.tsv")
anno.path <- file.path("~/Desktop/Thesis_Work/Projects/AD_WAS/Circos/AD_annotations_fin.tsv")

# pheno.path <-file.path("/project/ssverma_shared/projects/CardiacPET/REGENIE_SEX_STRAT_WITH_BATCH/Meta/Output/CMVD_ALL.global_reserve_merged.meta")
# anno.path <- file.path("/project/ssverma_shared/projects/CardiacPET/REGENIE_SEX_STRAT_WITH_BATCH/Summary/sumstats_all_replication_genes_CAD.csv")
# twas.path <- file.path("~/group/personal/rasika/CMVD_TWAS/collated_TWAS_results_withancestrypops_nopcutoff_GTEXpos.csv")

# 4 outputs
genes.out <- file.path("~/Desktop/Thesis_Work/Projects/AD_WAS/Circos/", "gene_track_data_ad")
manhattan.out <- file.path("~/Desktop/Thesis_Work/Projects/AD_WAS/Circos/", "manhattan_data_ad")
twas.out <- file.path("~/Desktop/Thesis_Work/Projects/AD_WAS/Circos/", "twas_data_ad")
pwas.out <- file.path("~/Desktop/Thesis_Work/Projects/AD_WAS/Circos/", "pwas_data_ad")

################################################################################
#                               loading zone                                   #
################################################################################

# #mega.data <- data.table::fread(mega.path)
# mega.data2 <- data.table::fread(pheno.path)
# colnames(mega.data2) <- gsub("#", "", colnames(mega.data2))

# Load the data for the GWAS annotation
gwas.data <- data.table::fread(gwas.path)
colnames(gwas.data) <- gsub("#", "", colnames(gwas.data))
#gene.data <- data.table::fread(gene.path)

anno.data2 <- data.table::fread(anno.path)
# colnames(anno.data2) <- gsub("#", "", colnames(anno.data2))
#anno.data <- readxl::read_excel(anno.path)

twas.data <- data.table::fread(twas.path)
pwas.data <- data.table::fread(pwas.path)
#rsid <- data.table::fread(rsid.path)



################################################################################
#                            format for gene track                             #
################################################################################

# Extract the global reserve data
genes_tp = anno.data2
# genes_tp$chr <- sub("^", "chr", genes_tp$CHR)
genes_tp$end = genes_tp$start + 1
# genes_tp$label = genes_tp$ANNOTATION
# genes_tp$group = genes_tp$PHENOTYPE
genes.bed.1 <- genes_tp[,c(2,3,6,4,5)]
# Threshold genes to be labeled that are above study-wide significance 
# genes_tp_c = rbind(genes_gr_AFR, genes_gr_EUR)
# genes_gr_c$group <- ifelse((genes_gr_c$COHORT == "CMVD_AFR" & genes_gr_c$P <= 0.0002), "AFR", 
#                            ifelse((genes_gr_c$COHORT == "CMVD_EUR" & genes_gr_c$P <= 0.0002), "EUR", "NS"))


# genes_new = 
#   mult = read.table("multiple_loci.txt")
# afr = read.table("afr_loci.txt")
# eur = read.table("eur_loci.txt")
# # Subset the data columns to have only
# # chr, start, end, label, group
# gene.bed <- format_genes(genes_gr_c) # 370

# 
# # Add additional important genes
# ADK = c("chr10", "74731160", "74731161", "ADK", "Important")
# CAPN2 = c("chr1", "223825378", "223825379", "CAPN2", "Important")
# ERC1 = c("chr12", "1447550", "1447551", "ERC1", "Important")
# AADAC = c("chr3", "151823495", "151823496", "AADAC", "Important")
# PLA2G5 = c("chr1", "20041185", "20041186", "PLA2G5", "Important")

# Create final formatted dataset
# genes.bed = rbind(gene.bed, ADK, CAPN2, ERC1, AADAC, PLA2G5)
genes.bed.1$start = as.numeric(genes.bed.1$start)
genes.bed.1$end = as.numeric(genes.bed.1$end)
genes.bed.1$chr <- ifelse(grepl("^chr", genes.bed.1$chr), genes.bed.1$chr, paste0("chr", genes.bed.1$chr))


save(genes.bed.1, file = genes.out)


################################################################################
#                      format for manhattan plot track                         #
################################################################################

mega.bed_gr <- format_manhattan(data = gwas.data,
                                annotations = anno.data2,
                                p.threshold = 0.5) # changed from 0.01 to 1 to include all SNPs
save(mega.bed_gr, file = manhattan.out)



################################################################################
#                              format for twas                              #
################################################################################
# Extract the global reserve data
# twas_gr = twas.data[grep("global_reserve", twas.data$phenotype),]
# p-value threshold
twas_gr = twas.data[twas.data$pvalue<=1,]

twas_gr4 = twas_gr[,c("gene_name", "Tissue", "pvalue", "gene_chr", "gene_start")]
twas_gr4$log10p <- -log10(twas_gr4$pvalue)
# Flip the orientation for the plot
twas_gr4$end = twas_gr4$gene_start + 1
# Make pretty colors
twas_gr4$tissue = ifelse(twas_gr4$Tissue=="Brain_Spinal_cord_cervical_c-1", "white", 
                         ifelse(twas_gr4$Tissue=="Brain_Cortex", "#CB4335",
                                ifelse(twas_gr4$Tissue=="Brain_Substantia_nigra", "#9B59B6",
                                       ifelse(twas_gr4$Tissue=="Colon_Sigmoid", "#4A235A",
                                              ifelse(twas_gr4$Tissue=="Liver", "#154360",
                                                     ifelse(twas_gr4$Tissue=="Vagina", "#5499C7",
                                                            ifelse(twas_gr4$Tissue=="Whole_Blood", "#85C1E9",
                                                                   ifelse(twas_gr4$Tissue=="Brain_Anterior_cingulate_cortex_BA24", "#0E6251",
                                                                          ifelse(twas_gr4$Tissue=="Brain_Frontal_Cortex_BA9", "#A2D9CE",
                                                                                 ifelse(twas_gr4$Tissue=="Brain_Hypothalamus", "darkgreen",
                                                                                        ifelse(twas_gr4$Tissue=="Brain_Caudate_basal_ganglia", "#D4AC0D",
                                                                                               ifelse(twas_gr4$Tissue=="Brain_Putamen_basal_ganglia", "#7D6608",
                                                                                                      ifelse(twas_gr4$Tissue=="Brain_Nucleus_accumbens_basal_ganglia", "#F39C12",
                                                                                                             ifelse(twas_gr4$Tissue=="Brain_Cerebellar_Hemisphere", "#D35400",
                                                                                                                    ifelse(twas_gr4$Tissue=="Brain_Putamen_basal_ganglia", "grey",
                                                                                                                           ifelse(twas_gr4$Tissue=="Brain_Cerebellum", "darkblue",
                                                                                                                                  ifelse(twas_gr4$Tissue=="Brain_Amygdala", "black",
                                                                                                                                         ifelse(twas_gr4$Tissue=="Brain_Hippocampus", "#641E16",""))))))))))))))))))


twas_gr5 = twas_gr4[,c(4,5,7,6,8)]

colnames(twas_gr5) <- c("chr", "start", "end", "value1", "value2")
twas_gr5$value1 <- twas_gr5$value1 * (-1)
twas_gr5 <- twas_gr5[order(-twas_gr5$value1), ]


save(twas_gr5, file = twas.out)

################################################################################
#                              format for pwas                              #
################################################################################
# Extract the global reserve data
# pheno = ""
# pwas_gr = pwas.data[grep(pheno, pwas.data$phenotype),]
# p-value threshold
pwas_gr = pwas.data[pwas.data$pvalue<=1,]
# pwas_gr$Model = "EA"
pwas_gr4 = pwas_gr[,c("gene_name", "Tissue", "pvalue", "gene_chr", "gene_start")]
pwas_gr4$log10p <- -log10(pwas_gr$pvalue)
# Flip the orientation for the plot
pwas_gr4$end_position = pwas_gr4$gene_start + 1
# Make pretty colors
pwas_gr4$tissue = ifelse(pwas_gr4$Tissue=="EA", "darkblue", 
                         ifelse(pwas_gr4$Tissue=="AA", "lightblue", ""))


pwas_gr5 = pwas_gr4[,c(4,5,7,6,8)]
colnames(pwas_gr5) <- c("chr", "start", "end", "value1", "value2")
pwas_gr5$value1 <- pwas_gr5$value1 * (-1)
pwas_gr5 <- pwas_gr5[order(-pwas_gr5$value1), ]


save(pwas_gr5, file = pwas.out)

