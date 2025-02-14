library(readxl)
library(showtext)
library(scales)
library(tikzDevice)
library(patchwork)
library(reshape2)
library(plyr)
library(latex2exp)
showtext_auto()
library(ggplot2)
font.add('Linux Libertine', regular = '/Library/Fonts/LinLibertine_R.otf',bold = '/Library/Fonts/LinLibertine_RB.otf')
options(
  tikzLatexPackages = c(
    getOption('tikzLatexPackages'),
    "\\usepackage{graphicx}"
  )
)
theme_Publication <- function(base_size=14, base_family="Linux Libertine") {
  library(grid)
  library(ggthemes)
  (theme_foundation(base_size=base_size, base_family=base_family)
    + theme(plot.title = element_text(face = "bold",
                                      size = rel(1.2), hjust = 0.5),
            text = element_text(),
            panel.background = element_rect(colour = NA),
            plot.background = element_rect(colour = NA),
            panel.border = element_rect(colour = NA),
            axis.title = element_text(face = "bold",size = rel(1)),
            axis.title.y = element_text(angle=90,vjust =2),
            axis.title.x = element_text(vjust = -0.2),
            axis.text = element_text(), 
            axis.line = element_line(colour="black"),
            axis.ticks = element_line(),
            panel.grid.major = element_line(colour="#C7C8C7", size=0.2),
            panel.grid.minor = element_line(colour="#f0f0f0", size=0),
            legend.key = element_rect(colour = NA),
            legend.position = "bottom",
            legend.direction = "horizontal",
            legend.key.size= unit(0.2, "cm"),
            legend.margin = margin(0, 0, 0, 0, "cm"),
            legend.title = element_text(face="italic"),
            plot.margin=unit(c(5,3,3,3),"mm"),
            strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0"),
            strip.text = element_text(face="bold")
    ))
  
}


log10_minor_break = function(...) {
  function(x) {
    minx = floor(min(log10(x), na.rm = T)) - 1;
    maxx = ceiling(max(log10(x), na.rm = T)) + 1;
    n_major = maxx-minx+1;
    major_breaks = seq(minx, maxx, by = 1)
    minor_breaks = 
      rep(log10(seq(1, 9, by = 1)), times = n_major) + 
      rep(major_breaks, each = 9)
    return(10^(floor(minor_breaks)))
  }
}
scal <- read_excel("./train_perform.xlsx", sheet = "train_perform")
scal <- melt(scal, id.vars = c("epoch"), measure.vars = c("train","val"),variable.name = "type", value.name = "loss")
# scal$size <- factor(scal$motif, levels = c("3-node motifs", "4-node motifs"))
# scal$name <- mapvalues(scal$motif, from = c("3-node motifs", "4-node motifs"
# ),
# to = c("3-node", "4-node"
# ))
# scal$type <- mapvalues(scal$type, from = c("train", "inference"),to = c("training time/epoch", "inference time"))
tikz(file = "model_analysis.tex", width = 4, height =3)
p_scal <- ggplot(scal, aes(x=epoch, y=loss,group=type, color=type,  shape = type)) +
  geom_line(linewidth=0.95)+
  geom_point(size=1.7) +
  geom_line(linewidth=0.95)+
  geom_point(size=1.7) +
  # scale_y_log10(
  #   labels = trans_format("log10", math_format(10^.x)),
  #   minor_breaks=log10_minor_break()) +
  # scale_x_log10(labels = trans_format("log10", math_format(10^.x)),
  #               minor_breaks=log10_minor_break()) +
  # facet_wrap(~ motif, ncol = 2) +
  # geom_abline(intercept = -0.05, slope = 0.48, linetype="dashed")+
  # annotate("text", x = 5000, y = 200, label =TeX("$f(x) = x + c$"), angle = 25, size = 2.5) +
  labs(y = "Huber Loss", x= "Epoch") + 
  theme_Publication() +
  scale_shape_manual(values=seq(0,15)) +
  theme(text = element_text(size = 9), axis.text = element_text(size = 9),
        legend.title = element_blank(), 
        legend.key.size = unit(1.2,"line"), legend.text=element_text(size=9),
        panel.spacing = unit(0.2 , "lines"),
        plot.margin = margin(t = 0.3, r = 0.5, b =0.2, l = 0.5, unit = "lines"))
print(p_scal)

dev.off()

