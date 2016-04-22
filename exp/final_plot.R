library(data.table)
library(ggplot2)
library(scales)
library(RColorBrewer)
library(gridExtra)
library(grid)

mytheme = theme_bw() + 
  theme(text = element_text(size = 22, family="Times"),
        strip.text = element_text(size = 22, family="Times", lineheight=0.2),
        strip.background = element_rect(fill="cornsilk", colour="black", size=1.0),
        axis.text.x = element_text(size = 18, family="Times"),
        axis.title.x = element_text(size = 20, family="Times", lineheight=0.1),
        axis.title.y = element_text(size = 20, family="Times", lineheight=0.1),
        axis.text.y = element_text(size = 18, family="Times"),
        legend.key=element_blank(),
        legend.title=element_blank(),
        legend.key.size=unit(2,"line"),
        legend.key.width=unit(4.8, "line"),
        legend.text = element_text(size = 22, family="Times"),
        panel.grid.minor = element_blank(),#line(colour="gray", linetype="solid", size=0.1),
        panel.grid.major = element_line(colour="gray", linetype="solid", size=0.5),
        panel.border = element_rect(color = "black", fill = NA, size = 1.0),
        panel.margin = unit(1.0, "line"),
        plot.title = element_blank())


algorithms = c("GP", "AFPO", "DFPO", "SNFPO", "ASNFPO")
problems = c("mod_quartic", "nonic", "keijzer4", "R1", "R2")
results_dir = "./gp_afpo_dfpo_snfpo_256_euclid_final"

read.file = function(file, problem, algorithm, xover) {
  data = read.csv(file)
  gens = seq(1, 1001, 10)
  data = data[gens,]
  data = subset(data, select="min_fitness")
  data$gen = gens
  data$problem = p
  data$alg = algorithm
  data$xover = xover
  return(data)
}

if (!exists("all.data")) {
  all.data = NULL
  for (p in problems) {
    print(p)
    for (a in algorithms) {
      print(a)
      pattern = paste("^", a, "_", p, sep="")
      files = list.files(path = results_dir, pattern = pattern, full.names = TRUE, ignore.case = TRUE)
      csvs = lapply(files, read.file, p, a, "standard")
      data = rbindlist(csvs)
      
      by=list(gen=data$gen, alg=data$alg, xover=data$xover, problem=data$problem)
      agg.data = aggregate(list(mean=data$min_fitness), by, FUN=mean)
      agg.data.sd = aggregate(data$min_fitness, by, FUN=sd)
      ci = 1.96 * (agg.data.sd$x / sqrt(100))
      agg.data$LL = agg.data$mean - ci
      agg.data$UL = agg.data$mean + ci
      
      pattern = paste("^", a, "_LGX_", p, sep="")
      files = list.files(path = results_dir, pattern = pattern, full.names = TRUE, ignore.case = TRUE)
      csvs = lapply(files, read.file, p, a, "geometric")
      data = rbindlist(csvs)
      
      by=list(gen=data$gen, alg=data$alg, xover=data$xover, problem=data$problem)
      agg.data.lgx = aggregate(list(mean=data$min_fitness), by, FUN=mean)
      agg.data.lgx.sd = aggregate(data$min_fitness, by, FUN=sd)
      ci = 1.96 * (agg.data.lgx.sd$x / sqrt(100))
      agg.data.lgx$LL = agg.data.lgx$mean - ci
      agg.data.lgx$UL = agg.data.lgx$mean + ci
      
      if (is.null(all.data)) {
        all.data = rbind(agg.data, agg.data.lgx)
      } else {
        all.data = rbind(all.data, agg.data, agg.data.lgx)
      }
    }
  }
}

problem_names <- c(
  `mod_quartic` = "\n QUARTIC \n",
  `nonic` = "\n NONIC \n",
  `keijzer4` = "\n KEIJZER-4 \n",
  `R1` = "\n R1 \n",
  `R2` = "\n R2 \n"
)

xover_names <- c(
  `standard` = "\n standard crossover \n",
  `geometric` = "\n geometric crossover \n"
)

all.data$xover <- factor(all.data$xover, levels = c("standard", "geometric"))
all.data$problem <- factor(all.data$problem, levels = c("mod_quartic", "nonic", "keijzer4", "R1", "R2"))
all.data$alg <- factor(all.data$alg, levels = c("GP", "AFPO", "DFPO", "SNFPO", "ASNFPO"))

palette = brewer.pal(5, "Set1")
palette[4] = palette[5]
palette[5] = palette[1]
palette[1] = "black"

plot = (ggplot(all.data) 
        + geom_line(aes(x=gen, y=mean, colour=alg, linetype=alg), size=1.5)
        + geom_ribbon(aes(x=gen, y=mean, ymin = LL, ymax = UL, fill=alg), alpha = 0.2)
        + facet_grid(problem ~ xover, labeller = labeller(problem = label_names, xover=xover_names))
        + scale_y_continuous(name="Fitness\n")
        + scale_x_continuous(name="\nGeneration")
        + scale_linetype_manual(values=c(3,1,5,1,6), labels=c("GP ", "AFPO ", "DFPO ", "ESNFPO ", "ASNFPO "))
        + scale_color_manual(values = palette, labels=c("GP ", "AFPO ", "DFPO ", "ESNFPO ", "ASNFPO "))
        + scale_fill_manual(values = palette, guide=FALSE)
        + coord_cartesian(ylim=c(0.0, 0.51))
        + mytheme
        + theme(legend.position="top"))

print(plot)
ggsave("R_final_plot.pdf", width = 10, height = 25)