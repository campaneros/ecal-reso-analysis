// ---------------------------------------------------------------------------
//  fit_fixedSC.C
//
//  Curva di risoluzione  sigma/E = N/E (+) S/sqrt(E) (+) C  con S e C FISSATI
//  ai valori misurati a 340 ohm, che e' la resistenza con piu' punti e col fit
//  meglio condizionato. S e C sono proprieta' del cristallo e della geometria;
//  la resistenza CATIA cambia il guadagno, cioe' N. Se e' cosi', a 400 e 500 ohm
//  deve bastare N libero.
//
//  Legge plot/root/points_resolution.csv, prodotto da plot/fit_fixedSC.py:
//      dataset,resistance,energy_nom,energy_true,sigma_over_E_pct,err_pct
//  dataset = runmean  media delle sigma per run
//            corr     come sopra ma con la risposta corretta evento per evento
//                     in (pos_eta, pos_phi), vedi plot/uniformita_pos.py
//
//  Scrive plot/root/resolution_fixedSC.root con, per ogni dataset e resistenza,
//  il TGraphErrors dei punti, la TF1 del fit libero, la TF1 del fit con S e C
//  congelati, e un TCanvas per dataset. In piu' un TTree `summary` coi parametri.
//
//  Uso, dalla radice del repo:
//      root -l -b -q plot/root/fit_fixedSC.C
//      root -l -b -q 'plot/root/fit_fixedSC.C("altro.csv","altro.root")'
// ---------------------------------------------------------------------------

#include <TGraphErrors.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TTree.h>
#include <TLegend.h>
#include <TPaveText.h>
#include <TAxis.h>
#include <TMath.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TString.h>
#include <TVirtualPad.h>
#include <TROOT.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <iostream>

// sigma/E in PERCENTO. N in GeV, S e C in percento: stessa convenzione di
// fit_plot.sh e di resolution_final.py.
double resoFun(double *x, double *p)
{
   double E = x[0];
   double n = 100.0 * p[0] / E;
   double s = p[1] / TMath::Sqrt(E);
   return TMath::Sqrt(n * n + s * s + p[2] * p[2]);
}

struct Points {
   std::vector<double> x, y, ex, ey;
};

void fit_fixedSC(const char *csvname = "plot/root/points_resolution.csv",
                 const char *outname = "plot/root/resolution_fixedSC.root")
{
   gStyle->SetOptStat(0);
   gStyle->SetOptFit(0);

   // ------------------------------------------------------------ lettura CSV
   std::ifstream in(csvname);
   if (!in.is_open()) {
      std::cout << "non riesco ad aprire " << csvname << std::endl;
      return;
   }
   std::map<std::string, Points> data;
   std::string line;
   std::getline(in, line);                       // intestazione
   while (std::getline(in, line)) {
      if (line.empty()) continue;
      std::stringstream ss(line);
      std::string ds, sR, sEn, sEt, sY, sE;
      std::getline(ss, ds,  ',');
      std::getline(ss, sR,  ',');
      std::getline(ss, sEn, ',');
      std::getline(ss, sEt, ',');
      std::getline(ss, sY,  ',');
      std::getline(ss, sE,  ',');
      if (sE.empty()) continue;
      std::string key = ds + "_" + sR;
      Points &p = data[key];
      p.x.push_back(std::atof(sEt.c_str()));
      p.y.push_back(std::atof(sY.c_str()));
      p.ex.push_back(0.0);
      p.ey.push_back(std::atof(sE.c_str()));
   }
   in.close();
   if (data.empty()) {
      std::cout << "nessun punto letto da " << csvname << std::endl;
      return;
   }

   TFile *fout = TFile::Open(outname, "RECREATE");
   if (fout == 0 || fout->IsZombie()) {
      std::cout << "non riesco a creare " << outname << std::endl;
      return;
   }

   // -------------------------------------------------------------- TTree
   int    t_dataset = 0;     // 0 = runmean, 1 = corr
   int    t_R = 0;
   int    t_mode = 0;        // 0 = free, 1 = S e C congelati a 340 ohm
   int    t_ndf = 0, t_npoints = 0;
   double t_N = 0, t_eN = 0, t_S = 0, t_eS = 0, t_C = 0, t_eC = 0, t_chi2 = 0;
   TTree *tree = new TTree("summary", "risultati del fit N/E (+) S/sqrt(E) (+) C");
   tree->Branch("dataset", &t_dataset, "dataset/I");
   tree->Branch("resistance", &t_R, "resistance/I");
   tree->Branch("mode", &t_mode, "mode/I");
   tree->Branch("N_MeV", &t_N, "N_MeV/D");
   tree->Branch("err_N_MeV", &t_eN, "err_N_MeV/D");
   tree->Branch("S_pct", &t_S, "S_pct/D");
   tree->Branch("err_S_pct", &t_eS, "err_S_pct/D");
   tree->Branch("C_pct", &t_C, "C_pct/D");
   tree->Branch("err_C_pct", &t_eC, "err_C_pct/D");
   tree->Branch("chi2", &t_chi2, "chi2/D");
   tree->Branch("ndf", &t_ndf, "ndf/I");
   tree->Branch("npoints", &t_npoints, "npoints/I");

   const int    nds = 2;
   const char  *dsname[nds] = {"runmean", "corr"};
   const int    nres = 3;
   const int    resist[nres] = {340, 400, 500};
   const int    colr[nres] = {kAzure + 2, kOrange + 7, kGreen + 2};

   std::cout << std::endl;
   printf("%8s %5s %8s %18s %9s %9s %14s\n",
          "dataset", "R", "mode", "N (MeV)", "S (%)", "C (%)", "chi2/ndf");

   for (int id = 0; id < nds; ++id) {

      // ---- 340 ohm libero: da qui escono S0 e C0
      std::string k340 = std::string(dsname[id]) + "_340";
      if (data.find(k340) == data.end()) {
         std::cout << "manca " << k340 << ", salto il dataset" << std::endl;
         continue;
      }
      Points &p340 = data[k340];
      TGraphErrors *g340 = new TGraphErrors((int)p340.x.size(), &p340.x[0], &p340.y[0],
                                            &p340.ex[0], &p340.ey[0]);
      TF1 *f340 = new TF1(Form("f_%s_340_free", dsname[id]), resoFun,
                          0.9 * (*std::min_element(p340.x.begin(), p340.x.end())),
                          1.05 * (*std::max_element(p340.x.begin(), p340.x.end())), 3);
      f340->SetParameters(0.3, 3.0, 0.3);
      f340->SetParLimits(0, 0.0, 5.0);
      f340->SetParLimits(1, 0.0, 20.0);
      f340->SetParLimits(2, 0.0, 5.0);
      g340->Fit(f340, "RQ0");
      double S0 = f340->GetParameter(1);
      double C0 = f340->GetParameter(2);
      delete g340;
      delete f340;

      TCanvas *c = new TCanvas(Form("c_%s", dsname[id]),
                               Form("resolution -- %s", dsname[id]), 1800, 600);
      c->Divide(3, 1);

      for (int ir = 0; ir < nres; ++ir) {
         int R = resist[ir];
         std::string key = std::string(dsname[id]) + "_" + std::to_string(R);
         if (data.find(key) == data.end()) continue;
         Points &p = data[key];
         int n = (int)p.x.size();
         double xmin = *std::min_element(p.x.begin(), p.x.end());
         double xmax = *std::max_element(p.x.begin(), p.x.end());

         TGraphErrors *g = new TGraphErrors(n, &p.x[0], &p.y[0], &p.ex[0], &p.ey[0]);
         g->SetName(Form("gr_%s_%d", dsname[id], R));
         g->SetTitle(Form("%d #Omega   %s;True beam energy [GeV];#sigma/E [%%]",
                          R, dsname[id]));
         g->SetMarkerStyle(20);
         g->SetMarkerSize(1.1);
         g->SetMarkerColor(colr[ir]);
         g->SetLineColor(colr[ir]);

         // ---- fit libero
         TF1 *ffree = new TF1(Form("f_%s_%d_free", dsname[id], R), resoFun,
                              0.9 * xmin, 1.05 * xmax, 3);
         ffree->SetParNames("N_GeV", "S_pct", "C_pct");
         ffree->SetParameters(0.3, 3.0, 0.3);
         ffree->SetParLimits(0, 0.0, 5.0);
         ffree->SetParLimits(1, 0.0, 20.0);
         ffree->SetParLimits(2, 0.0, 5.0);
         ffree->SetLineColor(kGray + 2);
         ffree->SetLineStyle(2);
         ffree->SetLineWidth(2);
         g->Fit(ffree, "RQ0");

         // ---- fit con S e C congelati ai valori di 340 ohm.
         // A 340 ohm non ha senso: S0 e C0 vengono da li'. Quindi solo 400 e 500.
         bool doFix = (R != 340);
         TF1 *ffix = new TF1(Form("f_%s_%d_fixedSC", dsname[id], R), resoFun,
                             0.9 * xmin, 1.05 * xmax, 3);
         ffix->SetParNames("N_GeV", "S_pct", "C_pct");
         ffix->SetParameters(0.3, S0, C0);
         ffix->SetParLimits(0, 0.0, 5.0);
         ffix->FixParameter(1, S0);
         ffix->FixParameter(2, C0);
         ffix->SetLineColor(kViolet);
         ffix->SetLineWidth(3);
         if (doFix) g->Fit(ffix, "RQ0");

         // ---- disegno
         c->cd(ir + 1);
         gPad->SetLogx();
         gPad->SetGridx();
         gPad->SetGridy();
         g->Draw("AP");
         ffree->Draw("same");
         if (doFix) ffix->Draw("same");

         TLegend *leg = new TLegend(0.42, 0.68, 0.90, 0.88);
         leg->SetBorderSize(1);
         leg->SetFillColor(0);
         leg->AddEntry(g, "#sigma/E", "lep");
         leg->AddEntry(ffree, "N, S, C free", "l");
         if (doFix) leg->AddEntry(ffix, "S, C fixed at 340 #Omega", "l");
         leg->Draw();

         TPaveText *pt = new TPaveText(0.42, 0.46, 0.90, 0.67, "NDC");
         pt->SetFillColor(0);
         pt->SetTextAlign(12);
         pt->SetTextFont(82);
         pt->SetTextSize(0.030);
         pt->AddText(Form("free   N %6.1f #pm %4.1f MeV", 1000 * ffree->GetParameter(0),
                          1000 * ffree->GetParError(0)));
         pt->AddText(Form("       S %6.3f  C %6.4f", ffree->GetParameter(1),
                          ffree->GetParameter(2)));
         pt->AddText(Form("       chi2/ndf %6.2f / %d", ffree->GetChisquare(),
                          ffree->GetNDF()));
         if (doFix) {
            pt->AddText(Form("fixed  N %6.1f #pm %4.1f MeV", 1000 * ffix->GetParameter(0),
                             1000 * ffix->GetParError(0)));
            pt->AddText(Form("       S %6.3f  C %6.4f  (fixed)", S0, C0));
            pt->AddText(Form("       chi2/ndf %6.2f / %d", ffix->GetChisquare(),
                             ffix->GetNDF()));
         }
         pt->Draw();

         // ---- salvataggio e stampa
         fout->cd();
         g->Write();
         ffree->Write();
         if (doFix) ffix->Write();

         for (int mode = 0; mode < (doFix ? 2 : 1); ++mode) {
            TF1 *f = (mode == 0) ? ffree : ffix;
            t_dataset = id;
            t_R = R;
            t_mode = mode;
            t_N = 1000 * f->GetParameter(0);
            t_eN = 1000 * f->GetParError(0);
            t_S = f->GetParameter(1);
            t_eS = f->GetParError(1);
            t_C = f->GetParameter(2);
            t_eC = f->GetParError(2);
            t_chi2 = f->GetChisquare();
            t_ndf = f->GetNDF();
            t_npoints = n;
            tree->Fill();
            printf("%8s %5d %8s %9.1f +- %4.1f %9.3f %9.4f %8.2f / %-3d\n",
                   dsname[id], R, (mode == 0 ? "free" : "fixedSC"),
                   t_N, t_eN, t_S, t_C, t_chi2, t_ndf);
         }
      }

      c->Update();
      fout->cd();
      c->Write();
   }

   fout->cd();
   tree->Write();
   fout->Close();
   std::cout << std::endl << "scritto " << outname << std::endl;
}
